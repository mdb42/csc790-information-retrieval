import os
import requests

DEFAULT_DATASETS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "gadsby": "https://www.gutenberg.org/files/47367/47367-0.txt",
    "bible": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "alice": "https://www.gutenberg.org/files/11/11-0.txt",
    "poe": "https://www.gutenberg.org/files/2147/2147-0.txt",
    "war_and_peace": "https://www.gutenberg.org/files/2600/2600-0.txt",
    "iliad": "https://www.gutenberg.org/files/6130/6130-0.txt",
    "grimm": "https://www.gutenberg.org/files/2591/2591-0.txt",
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "dracula": "https://www.gutenberg.org/files/345/345-0.txt",
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "sense_and_sensibility": "https://www.gutenberg.org/files/161/161-0.txt",
    "the_adventures_of_huckleberry_finn": "https://www.gutenberg.org/files/76/76-0.txt",
    "the_count_of_monte_cristo": "https://www.gutenberg.org/files/1184/1184-0.txt",
    "metamorphosis": "https://www.gutenberg.org/files/5200/5200-0.txt",
    "the_picture_of_dorian_gray": "https://www.gutenberg.org/files/174/174-0.txt",
    "the_time_machine": "https://www.gutenberg.org/files/35/35-0.txt",
    "ulysses": "https://www.gutenberg.org/files/4300/4300-0.txt",
    "the_rise_and_fall_of_the_third_reich": "https://www.gutenberg.org/files/17217/17217-0.txt",
    "the_divine_comedy": "https://www.gutenberg.org/files/8800/8800-0.txt",
    "the_art_of_war": "https://www.gutenberg.org/files/132/132-0.txt",
    "aeneid": "https://www.gutenberg.org/files/228/228-0.txt",
    "the_republic": "https://www.gutenberg.org/files/1497/1497-0.txt",
    "don_quixote": "https://www.gutenberg.org/files/996/996-0.txt",
    "the_trial": "https://www.gutenberg.org/files/7849/7849-0.txt",
    "maria_stuart": "https://www.gutenberg.org/files/16325/16325-0.txt",
    "emma": "https://www.gutenberg.org/files/158/158-0.txt",
    "the_holy_bible_kjv": "https://www.gutenberg.org/files/10/10-0.txt",
    "a_midsummer_nights_dream": "https://www.gutenberg.org/files/1514/1514-0.txt",
    "the_jungle_book": "https://www.gutenberg.org/files/236/236-0.txt",
    "siddhartha": "https://www.gutenberg.org/files/2500/2500-0.txt",
    "anna_karenina": "https://www.gutenberg.org/files/1399/1399-0.txt",
    "les_miserables": "https://www.gutenberg.org/files/135/135-0.txt",
    "the_three_musketeers": "https://www.gutenberg.org/files/1256/1256-0.txt",
    "the_stranger": "https://www.gutenberg.org/files/4935/4935-0.txt",
    "heart_of_darkness": "https://www.gutenberg.org/files/219/219-0.txt",
    "the_old_man_and_the_sea": "https://www.gutenberg.org/files/19222/19222-0.txt",
    "the_great_gatsby": "https://www.gutenberg.org/files/64317/64317-0.txt",
    "1984": "https://www.gutenberg.org/files/1342/1342-0.txt"
}

def download_and_chunk(dataset_key, output_dir, chunk_size=10000):
    if dataset_key not in DEFAULT_DATASETS:
        raise ValueError(f"No URL found for dataset '{dataset_key}'.")

    url = DEFAULT_DATASETS[dataset_key]

    print(f"Downloading '{dataset_key}' from {url}...")
    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading '{dataset_key}': {e}")
        return

    full_text = r.text

    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Determine existing chunks to avoid re-downloading
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(f"{dataset_key}_") and f.endswith(".txt")]
    if existing_files:
        print(f"'{dataset_key}' already has {len(existing_files)} chunks. Skipping download.")
        return

    # Chunk the text
    start = 0
    file_count = 1
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end]
        filename = f"{dataset_key}_{file_count}.txt"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk)

        start = end
        file_count += 1

    print(f"Saved {file_count - 1} chunks for '{dataset_key}' into '{output_dir}'.")

def main():
    if not os.path.exists('documents'):
            os.makedirs('documents')

    for dataset_key in DEFAULT_DATASETS.keys():
        download_and_chunk(dataset_key, output_dir="documents", chunk_size=5000)

if __name__ == "__main__":
    main()
