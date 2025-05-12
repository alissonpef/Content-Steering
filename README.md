# Content Steering with Reinforcement Learning: VM Simulation

**See it in action!** Watch a video demonstrating the project's functionality:
[▶️ Watch the Demo Video](https://www.youtube.com/watch?v=LHX0iUvxh3o&ab_channel=AlissonPereira%7CDev) 

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

    b.  **Run `app.py`:**
        Still in the project root directory (`content-steering/`). The `tutorial` password may be requested if the `tutorial` user does not have permission to access the Docker socket (usually resolved by adding the user to the `docker` group or running with `sudo`):
    ```bash
    sudo python3 steering-service/src/app.py
    ```
    You should see messages indicating the Flask server is running. Keep this terminal open.

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

### Generating Graphs (After Simulation)

1.  **Stop the Servers:** Press `Ctrl+C` in terminals 1 (Steering Service) and 2 (HTML Server).

2.  **Run the Graph Generation Script:**
    In the terminal, from the project root directory (`content-steering/`):
    ```bash
    python3 Generate_graphs.py
    ```
    The graph images will be saved in `~/Documents/content-steering/Files/Img/`. 

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