# AuctionHouse

## Setup (one time)

### 1. Deploy backend to Render
1. Push this repo to GitHub
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Set these values:
   - **Name:** auctionhouse
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click Deploy
6. Copy your Render URL (e.g. `https://auctionhouse.onrender.com`)

### 2. Update index.html with your Render URL
In `index.html`, replace `RENDER_URL_PLACEHOLDER` with your actual Render URL.
Then push again.

### 3. Enable GitHub Pages
1. Go to your GitHub repo → Settings → Pages
2. Set Source to `main` branch, `/ (root)` folder
3. Your site will be at `https://YOUR_USERNAME.github.io/AuctionHouse`

## Dev account
- Username: `adrianperez`
- Password: `xbox5678`
