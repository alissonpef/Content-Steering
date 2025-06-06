 Content Steering with Reinforcement Learning: VM Simulation

**See it in action!** Watch a video demonstrating the project's functionality:
[▶️ Watch the Demo Video](https://www.youtube.com/watch?v=DSD8DpCHHQM) 

This project demonstrates the application of Content Steering, using the DASH protocol and Reinforcement Learning (Epsilon-Greedy), to optimize cache server selection in a simulated video streaming environment. The testing and simulation environment is configured for execution within the provided VirtualBox VM.

## Virtual Machine (VM) Environment

*   **User:** `tutorial`
*   **Password:** `tutorial`

### Initial Environment and Project Setup

1.  **Download Pre-configured VM (Optional Base):**
    A VirtualBox VM with base software (Docker, Python, mkcert) is available. **Important: The project code in this VM is outdated.**
    [Download VM via Google Drive](https://drive.google.com/file/d/1mCB585muebdJIN6yXbioIoD1762svy3T/view?usp=sharing)

    If you choose to use this VM, **DO NOT use the project code that comes with it**. Follow the steps below to get the latest version.

2.  **Requirements for Using the VM:**
    *   Oracle VirtualBox installed on your system. More information: [https://www.virtualbox.org/](https://www.virtualbox.org/)

3.  **Getting the Latest Project Code in the VM:**
    After importing the VM and logging in (user: `tutorial`, password: `tutorial`):
    *   Open a terminal.
    *   We recommend cloning the latest version of the project repository directly into the VM. Navigate to a suitable directory (e.g., `~/Documents/`) and clone the repository:
    ```bash
    cd ~/Documents/
    git clone <https://github.com/alissonpef/Content-Steering> content-steering
    cd content-steering/
    ```
    This directory, `~/Documents/content-steering/`, will be referred to as the "project root directory".

4.  **Preparing the `dataset` Directory:**
    The video dataset ( `.mp4` files, `manifest.mpd`, etc.) is essential for the simulation.
    *   **If you used the pre-configured VM:** The outdated VM contains a `dataset` directory at `~/Documents/content-steering-tutorial/dataset/`.
    *   **Copy this `dataset` directory to the newly cloned project directory.**
        Assuming the new project was cloned into `~/Documents/content-steering/` and the old VM project is in `~/Documents/content-steering-tutorial/`:
    ```bash
    cp -r ~/Documents/content-steering-tutorial/dataset/ ~/Documents/content-steering/
    ```
    This ensures that the `content-steering/dataset/` directory contains the necessary video files for the cache servers.

## Step-by-Step Execution in the VM (with Updated Code)

Follow these instructions **inside the VirtualBox VM**, in the updated project root directory (e.g., `~/Documents/content-steering/`). Two terminals will be required.

### Terminal 1: Prepare and Start Backend Services

1.  **Navigate to the Project Root Directory:**
    (E.g., `cd ~/Documents/content-steering/`)

2.  **Generate and Place SSL Certificates:**
    SSL certificates are required to run services over HTTPS. The `create_certs.sh` script (located in the project root `content-steering/`) uses `mkcert` to generate locally trusted certificates.

    a.  **Execute the Certificate Creation Script:**
        From the project root directory (`content-steering/`), run:
    ```bash
    ./create_certs.sh video-streaming-cache-1 video-streaming-cache-2 video-streaming-cache-3 steering-service
    ```
    *Note: `mkcert` may request the user's password (`tutorial`) to install the local CA if it's not already installed.*

    b.  **Verification (Optional):**
        The `create_certs.sh` script should create and move certificates to the following directories within the project:
        *   **Cache Servers:** `content-steering/streaming-service/certs/` (files like `video-streaming-cache-1.pem`, `video-streaming-cache-1-key.pem`, etc.)
        *   **Steering Service:** `content-steering/steering-service/certs/` (files `steering-service.pem`, `steering-service-key.pem`)

3.  **Start Cache Servers and Configure Name Resolution (`/etc/hosts`):**
    The `starting_streaming.sh` script handles starting the cache server Docker containers and automatically updating the `/etc/hosts` file. This ensures proper name resolution for the services.
    ```bash
    sudo ./starting_streaming.sh
    ```
    After the script finishes, it will have:
    *   Started the `video-streaming-cache-1`, `video-streaming-cache-2`, and `video-streaming-cache-3` containers.
    *   Updated `/etc/hosts` to map these container names to their Docker IPs and `steering-service` to `127.0.0.1`.
    *You can verify the containers are running with `docker ps`.*

4.  **Start the Steering Service (Orchestrator):**
    a.  **Install Python Dependencies (Only the first time or if `requirements.txt` is changed):**
        From the project root directory (`content-steering/`). The `tutorial` password will be requested:
    ```bash
    sudo pip3 install -r steering-service/requirements.txt
    ```

    b.  **Run `app.py` specifying the desired steering strategy:**
        Still in the project root directory (`content-steering/`). The `tutorial` password may be requested.
        Choose **one** of the following commands to start the service:

    * **UCB1:**
    ```bash
        sudo python3 steering-service/src/app.py --strategy ucb1
    ```

    * **Epsilon-Greedy:**
    ```bash
        sudo python3 steering-service/src/app.py --strategy epsilon_greedy
    ```

    * **Random Selection:**
    ```bash
        sudo python3 steering-service/src/app.py --strategy random
    ```

    * **No Steering:**
    ```bash
        sudo python3 steering-service/src/app.py --strategy no_steering
    ```
    * **Optimal Strategy (Oracle Best Choice):**
    ```bash
    python3 steering-service/src/app.py --strategy oracle_best_choice
    ```
    * **Add Options:**
        *   `--verbose` or `-v` for detailed debug logs.
        *   `--log_suffix <suffix>` (e.g., `_testScenario1`) to append a suffix to log filenames for better organization.

    The Flask server will start (ideally on `https://0.0.0.0:30500`). Keep this terminal open.

### Terminal 2: Serve the Client Interface (HTML Player)

1.  **Navigate to the Project Root Directory:**
    (E.g., `cd ~/Documents/content-steering/`)

2.  **Start a Simple HTTPS Server for the HTML:**
    ```bash
    python3 -m http.server 8000
    ```
    Keep this terminal open.

### Running the Simulation in the Browser

1.  **Access the Player Interface:**
    In the VM's browser, go to: `http://127.0.0.1:8000/Content%20Steering.html`.

2.  **Load the MPD Manifest:**
    The default URL (`https://video-streaming-cache-1/Eldorado/4sec/avc/manifest.mpd`) should work. Click "**Load MPD**".

3.  **Configure and Start the Simulation:**
    *   Adjust the desired parameters in the "Simulation Setup" panel.
    *   Click "**Start Simulation**".

4.  **Data Collection:**
    During the simulation, the file `~/Documents/content-steering/Files/Data/simulation_log.csv` will be populated with simulation data. 


### Post-Simulation: Data Processing and Graph Generation

After running your simulations, individual log files (e.g., `log_<strategy_name>_<number>.csv` or `log_<strategy_name>_<suffix>_<number>.csv`) will be in `Graphics/Logs/`. The following steps guide you through processing these logs and generating various graphs. All graph-related scripts (`Generate_graphs.py`, `aggregate_logs.py`, `Generate_aggregated_graphs.py`, `analyze_server_choices.py`, `Generate_compare_graphs.py`) are located in the `Graphics/` directory.

**First, navigate to the `Graphics` directory from your project root:**
```bash
cd Graphics/
```

**Step 1: Aggregate Multiple Log Files for Each Strategy**
If you have run a particular strategy multiple times (resulting in files like `log_ucb1_1.csv`, `log_ucb1_2.csv`), use `aggregate_logs.py` to combine these into a single "average" log file. This script reads from `Graphics/Logs/` and saves the aggregated files into `Graphics/Logs/Average/`.

*   **To aggregate logs for each strategy (run from `Graphics/` directory):**
    ```bash
    python3 aggregate_logs.py ucb1
    python3 aggregate_logs.py epsilon_greedy
    python3 aggregate_logs.py random
    python3 aggregate_logs.py no_steering
    python3 aggregate_logs.py oracle_best_choice
    ```
    This will create files like `Graphics/Logs/Average/log_ucb1_average.csv`, `Graphics/Logs/Average/log_oracle_best_choice_average.csv`, etc. *(Note: The script adds a numeric suffix like `_1` if an aggregated file with the exact same name already exists).*
*   **If you used a `--log_suffix` (e.g., `_myTest`) for a set of runs:**
    ```bash
    python3 aggregate_logs.py ucb1 --suffix_pattern _myTest
    ```
    This creates, for example, `Graphics/Logs/Average/log_ucb1_myTest_average.csv`.

**Step 2: Generate Graphs from Aggregated Log Files**
Use `Generate_aggregated_graphs.py` to create a focused set of plots from a single *aggregated* `_average.csv` file. This provides an overview of each strategy's average performance.

*   **To process a specific aggregated log file (examples run from `Graphics/` directory):**
    ```bash
    # Example for an aggregated UCB1 log:
    python3 Generate_aggregated_graphs.py log_ucb1_average.csv
    
    # Example for an aggregated Epsilon-Greedy log:
    python3 Generate_aggregated_graphs.py Logs/Average/log_epsilon_greedy_average.csv # Example with explicit path

    # Example for an aggregated Optimal Strategy (Oracle Best Choice) log:
    python3 Generate_aggregated_graphs.py log_oracle_best_choice_average.csv
    ```
    *(The script searches `Graphics/Logs/Average/` by default if only the filename is given).*
    Graphs for each processed file will be saved in a corresponding subdirectory within `Graphics/Img/` (e.g., `Graphics/Img/log_ucb1_average/`).

**Step 3: Generate Comparative Analysis Table for Strategy Accuracy**
Use `analyze_server_choices.py` to generate a table comparing the accuracy of different strategies in choosing the dynamically optimal server. It processes all `*_average.csv` files in `Graphics/Logs/Average/` by default.

*   **Run the script (from `Graphics/` directory):**
    ```bash
    python3 analyze_server_choices.py
    ```
    The table will be printed to the console.
    *   A CSV file (default: `Graphics/Logs/Average/dynamic_best_choice_accuracy.csv`) will be saved.
    *   An image of the table (default: `Graphics/Img/analysis_tables/dynamic_best_choice_accuracy_table.png`) will be saved.
    *   Use `--output_csv <path>` and `--output_img <path>` for custom output locations.

**Step 4: Generate a Single Graph Comparing Average Latencies Across All Strategies**
After generating the `_average.csv` files for each strategy you wish to compare, use `Generate_compare_graphs.py`.

*   **To compare the default metric (`experienced_latency_ms` - chosen server latency based on oracle):**
    ```bash
    python3 Generate_compare_graphs.py
    ```
*   **To compare a different metric, e.g., client-measured latency:**
    ```bash
    python3 Generate_compare_graphs.py --metric experienced_latency_ms_CLIENT
    ```
*   **To compare the optimal possible latency (should be similar across strategies for the same environment):**
    ```bash
    python3 Generate_compare_graphs.py --metric dynamic_best_server_latency
    ```
    A single comparison graph (e.g., `all_strategies_latency_comparison.png`) will be saved directly in `Graphics/Img/`.

**(Optional) Step 5: Generate Detailed Graphs for Individual Simulation Runs**
Use `Generate_graphs.py` to create a comprehensive set of plots from a single, *non-aggregated* simulation log file. This helps analyze specific runs in detail.

*   **To process a specific individual log file (examples run from `Graphics/` directory):**
    ```bash
    # For an Epsilon-Greedy run:
    python3 Generate_graphs.py log_epsilon_greedy_1.csv 
    # or (if it automatically adds .csv and finds in Logs/)
    python3 Generate_graphs.py log_epsilon_greedy_1

    # For a UCB1 run:
    python3 Generate_graphs.py Logs/log_ucb1_1.csv # Example with explicit path

    # For an Optimal Strategy (Oracle Best Choice) run:
    python3 Generate_graphs.py log_oracle_best_choice_1.csv
    ```
*   If you run `python3 Generate_graphs.py` without arguments (while inside the `Graphics/` directory), it will attempt to process all non-aggregated `.csv` files found in `Graphics/Logs/`.

    Graphs for each processed file will be saved in a corresponding subdirectory within `Graphics/Img/` (e.g., `Graphics/Img/log_epsilon_greedy_1/`).

---

## Additional Useful Commands

*   **`docker ps`**
    *   Lists currently running Docker containers.
*   **`docker compose -f streaming-service/docker-compose.yml down`**
    *   (Run from the `content-steering/streaming-service/` directory)
    *   Stops and removes the cache server containers and networks defined in `docker-compose.yml`.
*   **`docker compose -f streaming-service/docker-compose.yml up -d`**
    *   (Run from the `content-steering/streaming-service/` directory)
    *   (Re)creates and (re)starts cache containers in detached mode (-d).
*   **`docker logs <container_name_or_id> -f`**
    *   Example: `docker logs video-streaming-cache-1 -f`
    *   Displays logs for a specific container in real-time (useful for debugging).
*   **`docker stop <container_name_or_id>`**
    *   Example: `docker stop video-streaming-cache-2`
    *   Stops a running container.
*   **`docker start <container_name_or_id>`**
    *   Example: `docker start video-streaming-cache-2`
    *   Starts a previously stopped container.

---