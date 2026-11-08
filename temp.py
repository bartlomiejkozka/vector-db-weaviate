import sys
from pathlib import Path
import h5py
import numpy as np

def inspect_h5_dataset(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"Błąd: Plik {file_path} nie istnieje!")
        sys.exit(1)

    print(f"--- Weryfikacja pliku: {path.name} ---")
    print(f"Rozmiar pliku na dysku: {path.stat().st_size / (1024**3):.2f} GB\n")

    with h5py.File(path, "r") as f:
        print("Dostępne klucze w pliku HDF5:")
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                print(f"  - '{key}': shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  - '{key}': [Grupa]")
        print("-" * 50)

        # 1. Sprawdzenie zbioru bazowego (Train)
        train_key = next((k for k in ["train", "base", "vectors"] if k in f), None)
        if train_key:
            train = f[train_key]
            n_rows, dim = train.shape
            print(f"✅ Baza wektorów ({train_key}):")
            print(f"   - Liczba wektorów: {n_rows:,}")
            print(f"   - Wymiarowość (D): {dim}")
            print(f"   - Typ danych: {train.dtype}")

            # Szybki test normalizacji na 5 pierwszych wektorach
            sample_vecs = train[:5]
            norms = np.linalg.norm(sample_vecs, axis=1)
            is_normalized = np.allclose(norms, 1.0, atol=1e-3)
            print(f"   - Znormalizowane (L2 = 1.0): {'TAK' if is_normalized else 'NIE (normy ~ ' + str(np.round(norms, 2)) + ')'}")
        else:
            print("❌ Brak zbioru treningowego/bazowego (szukano: train, base, vectors)!")

        # 2. Sprawdzenie zapytań (Test / Queries)
        test_key = next((k for k in ["test", "query", "queries"] if k in f), None)
        if test_key:
            test = f[test_key]
            print(f"\n✅ Zbiór zapytań ({test_key}):")
            print(f"   - Liczba zapytań: {test.shape[0]:,}")
            print(f"   - Wymiarowość: {test.shape[1]}")
            print(f"   - Typ danych: {test.dtype}")
        else:
            print("\n❌ Brak zbioru testowego/zapytań (szukano: test, query, queries)!")

        # 3. Sprawdzenie Ground Truth (Sąsiedzi)
        gt_key = next((k for k in ["neighbors", "ground_truth", "indices"] if k in f), None)
        if gt_key:
            gt = f[gt_key]
            print(f"\n✅ Ground Truth ({gt_key}):")
            print(f"   - Kształt: {gt.shape} (liczba zapytań x top-k sąsiadów)")
            print(f"   - Typ indeksów: {gt.dtype}")
            
            # Weryfikacja czy indeksy mieszczą się w zakresie bazy
            sample_gt = gt[:10]
            max_idx = np.max(sample_gt)
            min_idx = np.min(sample_gt)
            print(f"   - Przykładowe top-3 dla zapytania #0: {gt[0][:3]}")
            print(f"   - Zakres indeksów w próbce: min={min_idx}, max={max_idx}")
            
            if train_key and max_idx >= f[train_key].shape[0]:
                print(f"   ⚠️ OSTRZEŻENIE: Indeksy przekraczają rozmiar zbioru train!")
            else:
                print("   - Indeksy poprawne (wskazują na wiersze z train).")
        else:
            print("\n❌ Brak ground truth (szukano: neighbors, ground_truth, indices)!")

        # 4. Sprawdzenie odległości (Distances - opcjonalne)
        dist_key = next((k for k in ["distances", "dists"] if k in f), None)
        if dist_key:
            dist = f[dist_key]
            print(f"\nℹ️ Dystanse ({dist_key}):")
            print(f"   - Kształt: {dist.shape}, dtype={dist.dtype}")
            print(f"   - Przykładowe dystanse dla zapytania #0: {np.round(dist[0][:3], 4)}")

        print("-" * 50)
        print("Weryfikacja zakończona sukcesem.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python inspect_dataset.py <sciezka_do_pliku.h5>")
    else:
        inspect_h5_dataset(sys.argv[1])
