import nltk
from nltk.tokenize import word_tokenize
import os
import Levenshtein
nltk.download('punkt')

def search_similar_files(directory, query, threshold=0.2):
    matching_files = []
    for filename in os.listdir(directory):
        similarity = Levenshtein.ratio(query.lower(), filename.lower())
        if similarity >= threshold:
            matching_files.append((filename, similarity))
    return matching_files

# Example usage:
target_directory = "C:/Users/DELL/Desktop/python/data"
query = "exercise"
matching_files = search_similar_files(target_directory, query)

if matching_files:
    matching_files.sort(key=lambda x: x[1], reverse=True)
    print("Similar Files:")
    for file, similarity in matching_files:
        print(f"{file} (Similarity: {similarity:.2f})")
else:
    print("No similar files found.")
    
# def choose_file(matching_files):
#     if not matching_files:
#         print("No matching files found.")
#         return None
#     else:
#         print("Matching Files:")
#         for i, file in enumerate(matching_files, start=1):
#             print(f"{i}. {file}")

#         choice = int(input("Choose a file by entering its number: ")) - 1

#         if 0 <= choice < len(matching_files):
#             return matching_files[choice]
#         else:
#             print("Invalid choice.")
#             return None

# # Example usage:
# chosen_file = choose_file(matching_files)
# print("Chosen File:", chosen_file)
