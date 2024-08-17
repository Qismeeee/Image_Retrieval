import chromadb
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'Image_Retrieval/{ROOT}/train')))

# print(CLASS_NAME)


def read_img_from_path(path, size):
    img = Image.open(path).convert('RGB').resize(size)
    return np.array(img)


def folder_to_images(folder, size):
    list_dir = [os.path.join(folder, name) for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_img_from_path(path, size)
        images_path.append(path)
    images_path = np.array(images_path)
    return images_np, images_path

# Calculating image similarity


def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_size)


def get_l1_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            rates = absolute_difference(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum((data - query) ** 2, axis=axis_batch_size)


def get_l2_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            rates = mean_square_difference(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    # prevent division by zero
    return np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            rates = cosine_similarity(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def correlation_coefficient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query_mean**2))
    data_norm = np.sqrt(np.sum(data_mean**2, axis=axis_batch_size))

    return np.sum(query_mean * data_mean, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)


def get_correlation_coefficient_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            rates = correlation_coefficient(query, images_np)
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


# ----------------------------------------------------------------
# Use CLIP
embedding_function = OpenCLIPEmbeddingFunction()
# Trich xuất hình ảnh thành vector đặc trưng


def get_single_image_embedding(image):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)


def get_l1_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = absolute_difference(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_l2_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = mean_square_difference(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = cosine_similarity(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_correlation_cofficient_score(root_img_path, query_path, size):
    query = read_img_from_path(query_path, size)
    query_embedding = get_single_image_embedding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        print(f"Processing folder: {folder}")
        if folder in CLASS_NAME:
            path = os.path.join(root_img_path, folder)
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(
                    images_np[idx_img].astype(np.uint8))
                embedding_list.append(embedding)
            rates = correlation_coefficient(
                query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + "/" + filename
            files_path.append(filepath)
    return files_path


data_path = f'Image_Retrieval/{ROOT}/train'
files_path = get_files_path(path=data_path)


def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath)
        embedding = get_single_image_embedding(image=image)
        embeddings.append(embedding)
    collection.add(
        embeddings=embeddings,
        ids=ids
    )


iloveyou
# Create a Chroma Client
chroma_client = chromadb.Client()

# Create a collection
l2_collection = chroma_client.get_or_create_collection(
    name="12_collection", metadata={"HNSW_SPACE": "12"})


def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image_embedding(query_image)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results


def plot_results(query_path, ls_path_score, reverse=False):
    query_img = Image.open(query_path)

    # Sắp xếp danh sách theo score (L1 score), tùy chọn reverse
    ls_path_score.sort(key=lambda x: x[1], reverse=reverse)
    num_images = len(ls_path_score[:5]) + 1

    # Hiển thị hình ảnh truy vấn
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_images, 1)
    plt.imshow(query_img)
    plt.title("Query Image", fontsize=12)
    plt.axis('off')

    # Hiển thị top 5 hình ảnh giống nhất
    for i, (path, score) in enumerate(ls_path_score[:5], start=2):
        img = Image.open(path)
        plt.subplot(1, num_images, i)
        plt.imshow(img)
        plt.title(f"Score: {score:.2f}", fontsize=10)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.5)

    plt.show()


# root_img_path = f"Image_Retrieval/{ROOT}/train"
# query_path = f"Image_Retrieval/{
#     ROOT}/test/African_crocodile/n01697457_18534.JPEG"
# size = (448, 448)
# query, ls_path_score = get_l1_score(root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=False)

# root_img_path = f"Image_Retrieval/{ROOT}/train"
# query_path = f"Image_Retrieval/{
#     ROOT}/test/African_crocodile/n01697457_18534.JPEG"
# size = (448, 448)
# query, ls_path_score = get_correlation_coefficient_score(
#     root_img_path, query_path, size)
# plot_results(query_path, ls_path_score, reverse=False)


test_path = f'Image_Retrieval/{ROOT}/test'
test_files_path = get_files_path(path=test_path)
test_path = test_files_path[1]
l2_results = search(image_path=test_path,
                    collection=l2_collection, n_results=5)
plot_results(image_path=test_path, files_path=files_path, results=l2_results)

# Create a collection
cosine_collection = chroma_client.get_or_create_cosine_collection(
    name="Cosine_collection", metadata={HNSW_SPACE: "cosine"})
add_embedding(collection=cosine_collection, files_path=files_path)
