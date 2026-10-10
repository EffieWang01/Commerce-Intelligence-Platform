import base64
import io
from pathlib import Path

import numpy as np


class DisabledMultimodalRetriever:
    provider_name = "disabled"

    def search(self, query: str, image_data: str | None, limit: int) -> list[tuple[object, float]]:
        return []


def _decode_image(image_data: str):
    from PIL import Image
    encoded = image_data.split(",", 1)[1] if image_data.startswith("data:") else image_data
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


class ChineseClipMultimodalRetriever:
    """Chinese-CLIP retrieval with Qdrant storage and an in-memory fallback."""

    provider_name = "chinese-clip"

    def __init__(self, products: list, model_name: str = "ViT-B-16", qdrant_url: str = "",
                 qdrant_api_key: str = "", collection: str = "commerce_product_images",
                 sync: bool = False):
        import torch
        from cn_clip.clip import load_from_name

        self.torch = torch
        self.products_by_id = {product.item_id: product for product in products}
        self.image_products = [
            product for product in products
            if product.image_url and Path(product.image_url).is_file()
        ]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name(model_name, device=self.device)
        self.client = None
        self.collection = collection
        self.image_vectors = None

        if not self.image_products:
            return

        if qdrant_url:
            try:
                self._configure_qdrant(qdrant_url, qdrant_api_key, sync)
                return
            except Exception:
                self.client = None
        self._build_memory_index()

    def _encode_catalog(self):
        from PIL import Image

        if not self.image_products:
            return None
        images = [
            self.preprocess(Image.open(product.image_url).convert("RGB"))
            for product in self.image_products
        ]
        with self.torch.no_grad():
            vectors = self.model.encode_image(self.torch.stack(images).to(self.device))
            return vectors / vectors.norm(dim=-1, keepdim=True)

    def _build_memory_index(self) -> None:
        self.image_vectors = self._encode_catalog()

    def _configure_qdrant(self, url: str, api_key: str, sync: bool) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = QdrantClient(url=url, api_key=api_key or None)
        exists = client.collection_exists(self.collection)
        if exists and sync:
            client.delete_collection(collection_name=self.collection)
            exists = False
        vectors = None
        if not exists or sync:
            vectors = self._encode_catalog()
        if not exists:
            dimension = int(vectors.shape[1]) if vectors is not None else 512
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        if vectors is not None:
            points = [
                PointStruct(
                    id=index,
                    vector=vectors[index].float().cpu().tolist(),
                    payload={"item_id": product.item_id},
                )
                for index, product in enumerate(self.image_products)
            ]
            if points:
                client.upsert(collection_name=self.collection, points=points, wait=True)
        self.client = client

    def _query_vector(self, query: str, image_data: str | None):
        from cn_clip.clip import tokenize

        with self.torch.no_grad():
            if image_data:
                image = self.preprocess(_decode_image(image_data)).unsqueeze(0).to(self.device)
                vector = self.model.encode_image(image)
            else:
                vector = self.model.encode_text(tokenize([query]).to(self.device))
            return vector / vector.norm(dim=-1, keepdim=True)

    def search(self, query: str, image_data: str | None, limit: int) -> list[tuple[object, float]]:
        vector = self._query_vector(query, image_data)
        if self.client is not None:
            hits = self.client.query_points(
                collection_name=self.collection,
                query=vector[0].float().cpu().tolist(),
                limit=limit,
                with_payload=True,
            ).points
            return [
                (self.products_by_id[hit.payload["item_id"]], max(0.0, float(hit.score)))
                for hit in hits
                if hit.payload.get("item_id") in self.products_by_id
            ]
        if self.image_vectors is None:
            return []
        scores = (self.image_vectors @ vector.T).squeeze(1).float().cpu().numpy()
        order = np.argsort(-scores)[:limit]
        return [(self.image_products[index], max(0.0, float(scores[index]))) for index in order]
