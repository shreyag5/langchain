# This module contains utility classes and functions for interacting with MiimansaClinicalTextRetriever

import csv
import os

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MiimansaUtility:
    @staticmethod
    def is_direct_hit(query, metadata, direct_hit_model, direct_hit_threshold):
        """
        Checks z normalized cosine similarity between the query and questions in our qid2keyword

        Args:
            query (str): input query
            metadata (dict): Metadata containing embeddings
            direct_hit_model: SentenceTransformer model used to encode the query
            direct_hit_threshold (float): Threshold to determine if the query is a direct hit

        Returns:
            tuple: A tuple containing a boolean indicating if there's a direct hit and the QID of the direct hit if any
        """
        embedding_query = direct_hit_model.encode(query).reshape(1, -1)
        mean_vector = metadata["mean_vector"]
        variance_vector = metadata["variance_vector"]

        cosine_similarities = {
            qid: cosine_similarity(
                (embedding_query - mean_vector) / np.sqrt(variance_vector),
                (embedding_question - mean_vector) / np.sqrt(variance_vector),
            )[0][0]
            for qid, embedding_question in metadata["qid2emb"].items()
        }
        direct_hits = [
            qid
            for qid, score in cosine_similarities.items()
            if score > direct_hit_threshold
        ]

        if len(direct_hits) == 1:
            return True, direct_hits[0]
        return False, None

    @staticmethod
    def append_to_csv(csv_path, data_dict):
        """
        Appends data to a CSV file
        """
        intermediate_dir = os.path.dirname(csv_path)
        os.makedirs(intermediate_dir, exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=data_dict.keys())
                writer.writeheader()

        # Read existing headers from the CSV file
        with open(csv_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames

        # Check if all keys in the data dictionary are present in the existing fields
        missing_fields = set(existing_fields) - set(data_dict.keys())
        for field in missing_fields:
            # Populate missing fields in the data dictionary with empty values
            data_dict[field] = ""

        # Append the data to the CSV file
        with open(csv_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=existing_fields)
            writer.writerow(data_dict)

    # MetadataBuilder
    @staticmethod
    def prepare_metadata(data_path, direct_hit_model, output_path="./metadata.pkl"):
        """
        Create mappings for document vs generated question ids, along with embeddings of generated question
        Args:
            data_path (str): CSV file path containing columns 'generated_question' and 'context' ('QID' and 'ID' are optional)
        """
        data = pd.read_csv(data_path)
        if "ID" not in data.columns:
            data["ID"] = pd.factorize(data["context"])[0]
        if "QID" not in data.columns:
            data["QID"] = pd.factorize(data["generated_question"])[0]
        data["ID"] = data["ID"].astype(str)
        data["QID"] = data["QID"].astype(str)

        print("Creating mappings...")
        qid2id = data[["QID", "ID"]].set_index("QID")["ID"].to_dict()
        qid2ques = (
            data[["QID", "generated_question"]]
            .set_index("QID")["generated_question"]
            .to_dict()
        )
        id2context = data[["ID", "context"]].set_index("ID")["context"].to_dict()

        print("Computing embeddings...")
        model = SentenceTransformer(direct_hit_model)
        question_embeddings = model.encode(data["generated_question"].tolist())
        qid2emb = dict(
            zip(data["QID"], [i.reshape(1, -1) for i in question_embeddings])
        )

        mean_vector = np.mean(question_embeddings, axis=0)
        variance_vector = np.var(question_embeddings, axis=0)

        metadata = {
            "qid2id": qid2id,
            "id2context": id2context,
            "qid2ques": qid2ques,
            "qid2emb": qid2emb,
            "mean_vector": mean_vector,
            "variance_vector": variance_vector,
        }
        joblib.dump(metadata, output_path)
        print(f"Metadata saved at {output_path}")
