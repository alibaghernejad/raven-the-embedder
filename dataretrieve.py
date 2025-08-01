from fastembed import ImageEmbedding
from qdrant_client import QdrantClient, models

def retrieve_data(query_text, client, collection_name, dense_embedding_model, late_interaction_embedding_model, sparse_embedding_model):
    # query_vector = dense_embedding_model.query_embed(query_text)
    query_vector = sparse_embedding_model.query_embed(query_text)
    res = client.query_points(
        collection_name,
        query=next(query_vector),
        using="bm25",
        limit=10,
    with_payload=True,
    )
    return res


def retrieve_data_sparse(query_text, client, collection_name, dense_embedding_model, late_interaction_embedding_model, sparse_embedding_model):
    # query_vector = dense_embedding_model.query_embed(query_text)
    query_vector = next(sparse_embedding_model.query_embed(query_text))
    res = client.query_points(
        collection_name,
        # query=next(query_vector),
        query=models.SparseVector(**query_vector.as_object()),
        using="bm25",
        limit=10,
    with_payload=True,
    )
    return res

def retrieve_data_hybrid(query_text, client:QdrantClient, collection_name, dense_embedding_model, late_interaction_embedding_model, sparse_embedding_model):
    dense_query_vector = next(dense_embedding_model.query_embed(query_text))
    sparse_query_vector = next(sparse_embedding_model.query_embed(query_text))
    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using="bm25",
            limit=100,
        ),
    ]
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        with_payload=True,
        limit=10,
    )

    return results

def retrieve_data_rerank(query_text, client, collection_name, dense_embedding_model, late_interaction_embedding_model, sparse_embedding_model):
    dense_query_vector = next(dense_embedding_model.query_embed(query_text))
    sparse_query_vector = next(sparse_embedding_model.query_embed(query_text))
    late_query_vector = next(late_interaction_embedding_model.query_embed(query_text))
    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using="all-MiniLM-L6-v2",
            limit=100,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using="bm25",
            limit=30,
        ),
    ]
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=late_query_vector,
        using="colbertv2.0",
        with_payload=True,
        limit=10,
    )
    return results

def retrieve_data_all(query_text, client, collection_name, dense_embedding_model, late_interaction_embedding_model, sparse_embedding_model):
    dense_query_vector = next(dense_embedding_model.query_embed(query_text))
    sparse_query_vector = next(sparse_embedding_model.query_embed(query_text))
    late_query_vector = next(late_interaction_embedding_model.query_embed(query_text))
    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using="all-MiniLM-L6-v2",
            limit=30,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using="bm25",
            limit=20,
        ),
        models.Prefetch(
            query=late_query_vector,
            using="colbertv2.0",
            limit=10,
        ),
    ]
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        with_payload=True,
        limit=10,
    )
    return results


def retrieve_data_rerank(query_text, client, collection_name, dense_embedding_model, late_interaction_embedding_model, sparse_embedding_model):
    dense_query_vector = next(dense_embedding_model.query_embed(query_text))
    sparse_query_vector = next(sparse_embedding_model.query_embed(query_text))
    late_query_vector = next(late_interaction_embedding_model.query_embed(query_text))
    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using="all-MiniLM-L6-v2",
            limit=100,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_query_vector.as_object()),
            using="bm25",
            limit=30,
        ),
    ]
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=late_query_vector,
        using="colbertv2.0",
        with_payload=True,
        limit=10,
    )
    return results

def retrieve_relations(query_text, client, collection_name, dense_embedding_model):
    dense_query_vector = next(dense_embedding_model.query_embed(query_text))
    dense_query_vector = next(dense_embedding_model.query_embed(query_text))
    prefetch = [
        models.Prefetch(
            query=dense_query_vector,
            using="text_vector",
            limit=20,
        )
        # models.Prefetch(
        #     query=dense_query_vector,
        #     using="image_vector",
        #     limit=10,
        # ),
    ]
    results = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        with_payload=True,
        limit=10,
    )
    return results
