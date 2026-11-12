
# ShoppingKart 🛒

### Smart Shopping Cart & Retail Analytics System

ShoppingKart is an intelligent supermarket shopping system that simplifies in-store shopping through indoor mapping, optimized navigation, product scanning, and automated billing. It also provides retailers with analytics to understand sales patterns and improve product placement.

## ✨ Features

### Customer Features

- **Indoor Mapping:** Digital supermarket mapping using a grid-based floor plan and DINOv2 image embeddings.
- **Route Optimization:** A* pathfinding for shortest paths and Held–Karp dynamic programming to optimize product visit order.
- **Smart Cart:** Product browsing, cart management, and QR/barcode scanning using OpenCV.
- **Product Registration:** Add new products with details and store locations.
- **Automated Billing:** Generate order records and downloadable PDF receipts.
- **Smart Display:** ESP32-S3 with an OLED display for cart updates, route information, and checkout feedback.

### Retail Analytics

- Sales, revenue, order count, and average basket value.
- Best-selling products ranked by units sold or revenue.
- Zone-wise comparison of shopper time and sales.
- Store activity heatmap.
- Suggested product-placement locations.

*Analytics note: Shopper movement is currently simulated. Actual checkout records contribute to sales analysis; footfall and shopper-time metrics are estimates, not real-time tracking.*

## 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| Backend & UI | Python, Streamlit |
| Computer Vision | OpenCV, DINOv2, PyTorch |
| Route Optimization | A*, Held–Karp DP |
| Analytics | Pandas, Altair, Matplotlib |
| Data Storage | JSON, CSV, NumPy |
| Hardware | ESP32-S3, SH1106 OLED, NeoPixel |
| Communication | MicroPython, USB Serial |
| Deployment | Docker, Docker Compose |

## 📁 Project Structure

```text

ShoppingKart/
├── checkout/
│   ├── Dockerfile
│   ├── app.py
│   ├── analytics.py
│   ├── bill.py
│   ├── make_qr.py
│   ├── products.json
│   └── qr_codes.html
│
├── esp32/
│   └── screen.py
│
├── localization/
│   ├── Dockerfile
│   ├── app.py
│   ├── build_database.py
│   ├── cross_match.py
│   ├── database.py
│   ├── demo_template.html
│   ├── dino.py
│   ├── evaluate.py
│   ├── extract_frames.py
│   ├── make_demo.py
│   ├── query_location.py
│   ├── route.py
│   ├── similarity_matrix.py
│   └── store_map.json
│
├── .dockerignore
├── .gitignore
├── demo.html
├── docker-compose.yml
├── requirements.txt
└── README.md

```

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/christinawdc/ShoppingKart.git
cd ShoppingKart
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the applications**

Localization:

```bash
cd localization
streamlit run app.py
```

Checkout (in a separate terminal):

```bash
cd checkout
streamlit run app.py
```

**Alternatively, use Docker:**

```bash
docker compose up --build
```

| Application | URL |
|---|---|
| Localization | [http://localhost:8512](http://localhost:8512) |
| Checkout | [http://localhost:8513](http://localhost:8513) |

## 🔮 Future Improvements

- Real-time indoor cart localization.
- Wireless communication between cart and system.
- Live navigation instructions on the OLED display.
- Real customer footfall tracking.
- Inventory monitoring and demand forecasting.
- Payment gateway integration.

## 📌 Project Status

Core mapping, image embedding, route optimization, product scanning, cart management, local checkout, PDF billing, ESP32 display communication, and retail analytics are implemented.

Real-time localization, live footfall tracking, wireless cart communication, and payment gateway integration remain future improvements.

