# Plan: środowisko Weaviate pod Common Baseline (Milestone 1)

## Context

Repo zawiera prowizoryczny test (`src/`, `config.toml`, `docker-compose.yml`, `Makefile`, `db`), który idzie do wymiany. Celem milestone'u 1 jest gotowe, powtarzalne środowisko: kontener Weaviate + kontener klienta, pobieranie datasetów VIBE, schemat/DDL, ładowanie, indeks, kontrola poprawności i pierwszy goal `db 1` (KNN). Dzięki temu w milestone 2 zostaje tylko mierzenie (sweep ef → `search.csv`, klastrowanie).

Wymagania wspólne (Common Baseline, PDF) są twarde: datasety i reguły, 4 przypięte rdzenie, 8 GB RAM bez swapu, ścieżki `/data` `/db` `/results`, kolejność kroków, CLI `db <goal> <params>`, format `build.csv`/`search.csv`, przypięte wersje.

Rzecz, na którą mamy wpływ, to **konfiguracja Weaviate**: zmienne środowiskowe serwera oraz typ indeksu i jego parametry. Parametry dobiera się automatycznie z rozmiaru zbioru (N, d), z możliwością ręcznego nadpisania. Plik Gemini (`specyfikacja_konfiguracji_weaviate_pod_common_baseline.md`) traktujemy tylko pomocniczo, bo ma nieaktualną wersję 1.27, legacy API i błędy, np. brak wielkiej litery w nazwach kolekcji.

Decyzje użytkownika:
- Tryb timingu jest wybierany w configu: `async` (ingest_s i build_s osobno) albo `sync` (wyjątek DDL: jeden ingest_s, pusty build_s).
- Domyślnie tylko HNSW. Zbiór przekraczający budżet RAM jest domyślnie **odrzucany**. W configu da się wybrać fallback (HNSW+RQ/PQ/BQ/SQ albo HFresh).
- Estymator pamięci ma dwa tryby: prosty (`raw × 1.5`, reguła z PDF) i dokładny (wektory + krawędzie HNSW × bajty na połączenie).

## Docelowa struktura repo

Stare pliki do usunięcia: `src/`, `config.toml`, `docker-compose.yml`, `Makefile` i `db`. Wszystkie zostaną przepisane.

```
Dockerfile                  # obraz klienta (python:3.12-slim, przypięte zależności)
docker-compose.yml          # weaviate + download + benchmark (+ load/index/check)
.env.example                # CPU pinning / porty / wersje – lokalne nadpisania
config/
  weaviate.env              # zmienne środowiskowe SERWERA (statyczne)
  datasets.toml             # rejestr datasetów VIBE (url, sha256, N, d, distance)
  profiles.toml             # progi, estymator, profile indeksu, nadpisania per dataset
scripts/                    # cienkie wrappery wymagane przez baseline
  download.py  ddl.py  load.py  index.py  check.py  show_config.py  benchmark.py (stub M2)
src/wvbench/
  settings.py   # ładowanie TOML + env + CLI, rozwiązywanie precedencji
  sizing.py     # estymator pamięci + wybór profilu (auto)
  dataset.py    # odczyt HDF5 (train/test/neighbors, point_type, distance attr), chunking
  client.py     # połączenie (gRPC), wait-for-ready
  schema.py     # DDL: SCHEMA part / INDEX part
  ingest.py     # bulk load 50k/chunk, batch do Weaviate
  indexing.py   # czekanie na vectorQueueLength==0, pomiar build_s
  metrics.py    # timery, sampler peak RSS (Prometheus), rozmiar /db, sysinfo
  results.py    # zapis build.csv / search.csv + log JSON z pełną konfiguracją
  goals/knn.py  # goal 1
  cli.py        # `db <goal> <params...>`
db                          # entrypoint CLI w kontenerze (exec python -m wvbench.cli)
results/                    # build.csv, search.csv, runs/*.json (commitowane)
README.md
```

## 1. Deployment (docker-compose + Dockerfile)

- **`weaviate`**: `cr.weaviate.io/semitechnologies/weaviate:1.40.1` (przypięty). Ustawienia:
  - `cpuset: ${WEAVIATE_CPUS:-0-3}`, `mem_limit: 8g`, `memswap_limit: 8g`
  - `env_file: config/weaviate.env`
  - named volume `weaviate_db:/db`
  - healthcheck `/v1/.well-known/ready`
  - porty 8080, 50051 i 2112 (metryki) wystawione na host tylko opcjonalnie
- **Obraz klienta** (`Dockerfile`): `python:3.12-slim` z `requirements.txt` przypiętym przez `==` (weaviate-client, h5py, numpy, requests). Wersje zamrażamy przy implementacji.
- **`download`**: `/data` montowany rw, uruchamia `scripts/download.py <dataset>`.
- **`benchmark`**: `/data:ro`, `weaviate_db:/db:ro` (do pomiaru rozmiaru), `./results:/results`. Ustawienia `cpuset: ${CLIENT_CPUS:-4}`, `depends_on: weaviate (healthy)`, entrypoint `db`. Komendy `load`, `index` i `check` to ten sam obraz z innym entrypointem (`scripts/*.py`).
- Komendy zgodne z PDF:
  - `docker compose build`
  - `docker compose run --rm download <ds>`
  - `docker compose run --rm benchmark db 1 443321 4`
- W README trzeba zanotować, że na macOS VM Dockera potrzebuje ≥5 vCPU i ~10 GB RAM. `WEAVIATE_CPUS` i `CLIENT_CPUS` można nadpisać w `.env`.

## 2. Zmienne środowiskowe serwera — `config/weaviate.env`

Te wartości są statyczne, bo żadna nie zależy od datasetu. Wszystko, co zależy od datasetu, jest ustawieniem kolekcji (pkt 3).

| Zmienna | Wartość | Dlaczego |
|---|---|---|
| `PERSISTENCE_DATA_PATH` | `/db` | ścieżka z baseline |
| `GOMAXPROCS` | `4` | = przypięte rdzenie (indeks budowany ≤4 wątkami) |
| `GOMEMLIMIT` | `6800MiB` | ~85% z 8 GiB, docs zalecają 80–90%; GC zaczyna działać przed OOM-killem |
| `LIMIT_RESOURCES` | `false` | inaczej nadpisuje GOMEMLIMIT i bierze cores-1 |
| `ASYNC_INDEXING` | `true` | potrzebne w trybie `async`; w trybie `sync` compose dostaje `false` przez `.env` (pkt 5) |
| `DEFAULT_VECTORIZER_MODULE` / `ENABLE_MODULES` | `none` / `` | czyste embeddingi, bez modułów |
| `DEFAULT_QUANTIZATION` | `none` | kompresja tylko jawnie z profilu |
| `QUERY_MAXIMUM_RESULTS` | `10000` | k=100 z zapasem |
| `PERSISTENCE_HNSW_MAX_LOG_SIZE` | `1GiB` | docs zalecają wartość zbliżoną do rozmiaru grafu (~0.3–0.7 GB u nas) |
| `PROMETHEUS_MONITORING_ENABLED` | `true` | pomocniczy podgląd RSS procesu Weaviate podczas load/index; nigdy nie odpytywany w trakcie search (pkt 5) |
| `DISABLE_TELEMETRY`, `LOG_LEVEL=warning`, `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true`, `CLUSTER_HOSTNAME=node1` | | redukcja szumu |

## 3. Konfiguracja indeksu: `config/profiles.toml` + `sizing.py`

### Rejestr datasetów (`datasets.toml`)

Dla każdego datasetu z PDF (glove-200, landmark-768, arxiv-768, celeba-2048, inaturalist-2048, hotpotqa-640) rejestr przechowuje:
- `url` (HF `vector-index-bench/vibe`)
- oczekiwane `rows`, `dim`, `distance`
- `sha256`, uzupełnione po pierwszym pobraniu

Rzeczywiste N, d, `point_type` i `distance` zawsze czytamy z atrybutów HDF5. Rejestr służy tylko do walidacji.

### Estymator pamięci (`[estimator]`)

```toml
[estimator]
mode = "detailed"            # "simple" | "detailed"
memory_limit_gb = 8.0
simple_factor = 1.5          # reguła baseline: raw*1.5 <= limit
bytes_per_link = 10          # docs Weaviate: ~8–10 B/połączenie (konfigurowalne do 20)
overhead_factor = 1.25       # GC, kopie, alokator
budget_fraction = 0.85       # porównujemy do GOMEMLIMIT, nie do 8 GB
```

- **simple**: `N·d·4 × 1.5 ≤ memory_limit`
- **detailed**: `(N·d·4 / compression + N · 2·maxConnections · bytes_per_link) × overhead ≤ memory_limit × budget_fraction`
  - Warstwa 0 HNSW ma 2·M połączeń. `compression` wynika z quantizera (none=1, sq/rq8=4, rq4=8, bq/rq1=32, pq=4·d/segments).
- Wynik estymatora, czyli szacunek w MB, wybrana reguła i decyzja, trafia do logu runu.

### Wybór profilu (reguły, pierwsza pasująca wygrywa)

Kryteria to `max_raw_gb`, `min_dim` i `max_dim`, więc liczy się zarówno N, jak i d. Wartości startowe niżej są do strojenia empirycznie w M2:

| Profil | Warunek | index | M (maxConnections) | efConstruction | ef sweep (M2) |
|---|---|---|---|---|---|
| `small` | raw ≤ 1.5 GB | hnsw | 48 | 256 | 16…512 |
| `medium` | raw ≤ 3 GB | hnsw | 32 | 192 | 16…512 |
| `large` | raw ≤ 5.34 GB | hnsw | 24 | 128 | 16…512 |
| `oversize` | estymator przekracza budżet | wg `oversize.policy` | | | |

```toml
[oversize]
policy = "reject"            # "reject" (domyślnie) | "fallback"
fallback = "hnsw_rq8"        # nazwa profilu: hnsw_rq8 | hnsw_rq4 | hnsw_pq | hnsw_bq | hnsw_sq | hfresh
```

Profile fallback są zdefiniowane w tym samym pliku:
- `hnsw_rq8`: `quantizer={type="rq", bits=8, rescore_limit=20}`
- `hnsw_pq`: `segments=d/8`, `training_limit=100000`
- `hfresh`: RQ 1-bit wymuszony przez Weaviate, `max_posting_size_kb=48`, `replicas=4`; parametrem sweepu jest `search_probe`

Po wybraniu fallbacku estymator jest liczony ponownie z kompresją. Jeśli dalej się nie mieści, run się wywala. `--allow-oversize` pozwala obejść tę kontrolę.

Dodatkowo jest profil `flat` (brak indeksu, brute force). PDF wymienia go jako baseline, a przyda się, jeśli prowadzący zabroni indeksów trzymanych w RAM. Można go wybrać ręcznie przez `--profile flat`.

**Wartości domyślne `[defaults]`** dziedziczy każdy profil. Wszystkie da się nadpisać per profil, per dataset (`[overrides.<ds>]`), przez `WVB_*` albo `--param`; nic nie jest zaszyte w kodzie.
- `distance = "auto"`: z pliku, a „any” → cosine
- `quantizer = "none"`
- `vector_cache_max_objects = 1e12`: cały zbiór w RAM. Mniejsza wartość, np. 0.3·N, wymusza czytanie części wektorów z dysku. To scenariusz „nie wszystko w RAM”, jeśli prowadzący tego zażąda.
- `flat_search_cutoff = 40000`: bez znaczenia, bo nie mamy filtrów
- `ef = -1`: dynamiczne ef liczone przez Weaviate z `dynamicEfMin/Max/Factor` i limitu k. `ef` jest mutowalne, więc w benchmarku (M2) przed każdym punktem sweepu robimy `collection.config.update(... hnsw(ef=X))`, co nadpisuje -1. Po sweepie przywracamy -1. Przy implementacji trzeba sprawdzić, czy Weaviate bierze `max(ef, k)` dla k=100; jeśli tak, sweep dla k=100 zaczyna się od 100.

### Precedencja (od najwyższej)

`--param key=value` (CLI) > zmienne `WVB_*` > `[overrides.<dataset>]` w `profiles.toml` > `--profile <name>` > auto-wybór.

`scripts/show_config.py <dataset>` wypisuje rozwiązaną konfigurację, estymację i powód wyboru bez dotykania bazy.

## 4. Schemat / DDL — `src/wvbench/schema.py` (+ `scripts/ddl.py`)

- Nazwa kolekcji to plik bez rozszerzenia, `-`→`_`. **Weaviate wymusza wielką pierwszą literę**, więc `glove_200_cosine` → `Glove_200_cosine`. Mapowanie opisujemy w README.
- Kolumna `id` z baseline: `id` w Weaviate jest zarezerwowane (UUID), więc:
  - UUID obiektu = `uuid.UUID(int=row)`. Mapowanie jest odwracalne i daje „ids only” bez czytania property.
  - property `row_id: INT` (int64, `index_filterable=True`, `index_searchable=False`) to jawna wymagana kolumna. Steruje nią `[schema] store_row_id = true` (domyślnie).
    - `false` oznacza tylko UUID: brak property, a id liczone z UUID.
    - Przy `true` koszt jest mały: ~8 B na obiekt plus bitmapa filtrowalna, czyli dziesiątki MB. Lekko wydłuża ingest i spełnia wymaganie „kolumny id” dosłownie.
    - Zapytania, check i goal 1 **zawsze** używają UUID (`return_properties=[]`), więc `row_id` nie wpływa na latencję.
- Kolumna `vector`: named vector `Configure.Vectors.self_provided(name="vector", vector_index_config=...)`. Jeśli nazwa „vector” okaże się zarezerwowana, używamy `"embedding"` i zapisujemy to w README.
- `sharding desired_count=1`, `replication factor=1`.
- `scripts/ddl.py` ma dwie oznaczone części: `# === SCHEMA PART ===` (drop + create, idempotentne) i `# === INDEX PART ===` (parametry indeksu z profilu). Ponieważ indeks w Weaviate jest częścią definicji kolekcji, INDEX PART jest stosowany przy create, a README to wyjaśnia (pkt 5). README dostaje też tabelę kolumn (`row_id` int64 / `vector` float32[d]).

## 5. Pipeline: download → schema → load → index → check

1. **download.py**: streamuje do `/data/<name>.hdf5.part`, a po pobraniu robi rename. Wypisuje rozmiar i sha256 i porównuje z `datasets.toml`. Istniejący plik tylko weryfikuje.
2. **ddl.py (schema)**: drop i create według rozwiązanego profilu.
3. **load.py**:
   - czyta `train` po 50 000 wierszy
   - konwertuje `uint8`→`float32`, jeśli `point_type` tego wymaga; poza tym nic nie zmienia
   - tryb wysyłki wybierany w `profiles.toml [ingest] mode`:
     - `client_fixed` (domyślnie, zgodnie z baseline): `batch.fixed_size(batch_size, concurrent_requests)`
     - `client_dynamic`: `batch.dynamic()`, klient sam dobiera rozmiar wg obciążenia serwera
     - `server_side`: server-side batching. Klient otwiera strumień, a serwer sam mówi, ile obiektów chce dostać (backpressure). Dokładne API sprawdzimy przy implementacji pod przypiętą wersję klienta i serwera (np. `client.batch.experimental()` / `stream()`). Jeśli wersja go nie wspiera, run kończy się czytelnym błędem.
     - We wszystkich trybach plik czytamy po 50 000 wierszy, a tryb trafia do logu runu.
   - po ostatnim chunku sprawdza `failed_objects` i przy błędach kończy się non-zero exit
   - mierzy `ingest_s` (perf_counter) i peak RSS
4. **index.py**, zależnie od `timing.mode` w `profiles.toml`:
   - `async` (ASYNC_INDEXING=true): `build_s` = od końca load do momentu, gdy suma `vectorQueueLength` == 0 i status shardów ≠ INDEXING (`client.cluster.nodes(output="verbose")`, poll co 1 s). README zaznacza, że budowa grafu startuje już w trakcie ładowania.
   - `sync` (ASYNC_INDEXING=false): wyjątek DDL, `ingest_s` = load+index, `build_s` puste.
   - Spójność wymusza `settings.py`: tryb z configu musi zgadzać się z `ASYNC_INDEXING` zwracanym przez serwer (`/v1/meta` lub sprawdzenie env). Przy niezgodności dostajemy błąd z instrukcją przełączenia `.env`.
5. **check.py**:
   - `count == N` (`aggregate.over_all`)
   - wymiar wektora == d
   - 10 losowych id (seed 42) przez `fetch_object_by_id(include_vector=True)` musi dać `np.allclose` z plikiem
   - non-zero exit przy błędzie
6. Na końcu zapisujemy wiersz `results/build.csv` (`kmeans_s` puste) oraz `results/runs/<ts>_<dataset>.json` z pełną konfiguracją: profil, env serwera, wersje, CPU, RAM, OS, data. Logowania tych danych wymaga PDF.

**Pomiary pomocnicze (`metrics.py`)**:
- **peak memory, źródło główne:** cgroup kontenera Weaviate.
  - Czytamy `/sys/fs/cgroup/memory.peak` (cgroup v2), a przy cgroup v1 fallback na `memory.max_usage_in_bytes`. To jest „container memory peak” z PDF.
  - Kontener Weaviate jest restartowany przed `load`, więc peak liczy się od zera. Ponieważ peak jest monotoniczny, odczyty po load, po index i po search dają peak dla kolejnych faz.
  - Odczyt robimy przez `docker exec weaviate cat …`: z hosta (Makefile/`scripts/mem_peak.sh`) albo z klienta przez Docker SDK z zamontowanym `/var/run/docker.sock`. Źródło ustawia `[metrics] memory_source = "cgroup" | "prometheus" | "off"`.
- **Prometheus (pomocniczo):** sampler co 0.5 s (`process_resident_memory_bytes`) działa **tylko w trakcie load/index**, gdzie narzut jest pomijalny. W trakcie search jest **wyłączony**, żeby nie fałszować p99, i robimy tylko odczyt przed i po.
- `db_mb`: `du` na `/db` (ro)
- `index_mb`: suma plików HNSW (`*/main.hnsw.commitlog.d` + snapshoty) w katalogu kolekcji

## 6. CLI — `db <goal> <params...>`

- `db 1 <query_id> <k>`: KNN dla wektora `test[query_id]`. Na stdout jedna linia na wynik w formacie `id distance`, od najbliższego, z `return_properties=[]`, a `id` liczony z UUID. Jakiekolwiek błędy idą na stderr z exit≠0.
- Dodatkowe flagi (`--dataset`, `--ef`) są opcjonalne. Domyślny dataset pochodzi z `WVB_DATASET`.
- Dispatcher jest gotowy na kolejne goale (klastry, M2).

## 7. Makefile + README

- `Makefile` to cienka nakładka na `docker compose`: `build`, `up`, `download DS=…`, `load DS=…`, `index`, `check`, `show-config DS=…`, `db Q="1 0 10"`, `mem-peak`, `clean`.
- README pokrywa: checklistę z zadania, komendy krok po kroku, tabelę kolumn, opis DDL i trybów timingu, tabelę zmiennych serwera, opis profili i estymatora oraz jak nadpisać parametry.

## Otwartość na zmiany po dyskusji na zajęciach

Niepewne rzeczy to RAM vs dysk, rezygnacja z indeksów i mniej metryk. Każda z nich to zmiana configu, nie kodu:
- „nie wszystko w RAM”: `vector_cache_max_objects` < N, albo profile `hnsw_rq*`, `hnsw_pq` lub `hfresh` (skompresowane w RAM, pełne wektory na dysku). Weaviate nie ma DiskANN, a HFresh jest jego dyskowo zorientowanym odpowiednikiem.
- „bez indeksu”: profil `flat`
- inne limity: `memory_limit_gb` w `[estimator]` i `mem_limit` w `.env`
- mniej metryk: kolumny CSV, których nie mierzymy, zostają puste

## Weryfikacja end-to-end

1. `docker compose build && docker compose up -d weaviate`. Sprawdzamy ready, a `docker inspect` pokazuje cpuset 0-3, Memory 8 GiB i MemorySwap 8 GiB.
2. `scripts/show_config.py` dla wszystkich 6 datasetów (bez bazy):
   - glove → small, landmark → medium, arxiv i inaturalist → large
   - hotpotqa → reject w obu trybach estymatora
   - z `policy=fallback` hotpotqa → hnsw_rq8 albo hfresh z przeliczonym budżetem
   - testy jednostkowe `sizing.py` (pytest) na tych przypadkach
3. `docker compose run --rm download glove-200-cosine` wypisuje rozmiar i sha256.
4. Pełny pipeline na glove (najmniejszy) w trybie `async`, a potem raz w `sync`. Wiersz `build.csv` ma oczekiwane puste i wypełnione pola, a `check` przechodzi.
5. `docker compose run --rm benchmark db 1 0 10` daje 10 linii `id distance`. Porównujemy z `neighbors[0][:10]` (recall bliski 1 przy ef domyślnym). `db 1 999999999 10` zwraca exit≠0.
6. Na koniec arxiv (największy zaakceptowany): peak RSS < 8 GB, brak OOM-kill (`docker inspect … OOMKilled=false`).
