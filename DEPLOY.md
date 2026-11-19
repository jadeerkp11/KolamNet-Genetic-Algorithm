# Putting Kolam Studio on a public URL

`localhost:8501` only exists on your own computer. To let other people use the app, it has to run on a server
that is on the internet. Pick ONE option below. All of them use the files already in this folder
(`requirements.txt`, `.streamlit/config.toml`, `Dockerfile`, `render.yaml`).

> Note: on a free host the app may "sleep" when unused and take ~30–60 s to wake up. Each click on
> *Create* runs the genetic algorithm on the host's CPU, so lower the population/generations under
> **Advanced** if many people use it at once.

---------------------------------------------------------------------------------------------------
## Option A — Streamlit Community Cloud (easiest, free, no Docker)
1. Create a free account at https://github.com and make a new **public** repository (e.g. `kolam-studio`).
2. Upload the contents of this folder to it (do NOT upload the `.venv` folder — `.gitignore` already skips it).
   Keep `app.py` and `requirements.txt` in the repository root.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Click **Create app** → choose your repository and branch → set **Main file path** to `app.py`.
5. Under **Advanced settings** choose Python **3.12** (or 3.11) and click **Deploy**.
6. After a minute you get a public link like `https://your-name-kolam-studio.streamlit.app`.

## Option B — Hugging Face Spaces (free, uses the Dockerfile)
1. Create an account at https://huggingface.co → **New Space** → SDK: **Docker** → Blank.
2. Upload all files of this folder to the Space.
3. Put this block at the very top of the Space's `README.md`:
   ```
   ---
   title: Kolam Studio
   emoji: 🪷
   sdk: docker
   app_port: 8501
   ---
   ```
4. The Space builds automatically and gives you a public URL.

## Option C — Render.com (free tier, uses `render.yaml`)
1. Push this folder to a GitHub repository.
2. At https://render.com choose **New → Blueprint**, select the repository.
3. Render reads `render.yaml`, builds the Dockerfile and publishes a `https://kolam-studio.onrender.com`-style URL.

## Option D — Quick temporary public link from your own PC (no account needed for a demo)
Good for showing the project to someone for an hour. The link works only while your PC and the app are running.
1. Start the app as usual (`run_windows.bat`).
2. Install Cloudflare's free tunnel tool (`cloudflared`) from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
3. In a second terminal run:
   ```
   cloudflared tunnel --url http://localhost:8501
   ```
4. It prints a `https://something.trycloudflare.com` address — share that.

---------------------------------------------------------------------------------------------------
## Test the Docker image locally (optional)
```
docker build -t kolam-studio .
docker run -p 8501:8501 kolam-studio
```
Open http://localhost:8501
