# Virtual Try-On Flask Application with Real Imagen 3 AI

A sophisticated web-based virtual try-on system featuring **Real Google Cloud Imagen 3 AI integration** with the complete pipeline: **Zero-shot Object Detection**, **ROI Key Points**, **Segmentation Mask**, and **Real AI Generate & Inpaint**.

## 🤖 **Real AI Integration**

This application now supports **Real Imagen 3 API** for photorealistic clothing generation and inpainting!

### **Two Modes Available:**

1. **🤖 Real Imagen 3 Mode**: Uses Google Cloud AI for production-quality results
2. **🎭 Enhanced Mock Mode**: Advanced simulation when AI APIs aren't configured

## 🚀 **Quick Setup with Real AI**

### **Option A: Full AI Setup (Recommended)**

#### 1. Google Cloud Setup

```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

#### 2. Enable APIs

```bash
# Enable required Google Cloud APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable ml.googleapis.com
```

#### 3. Set Environment Variables

```bash
# Set your Google Cloud project
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

#### 4. Install Dependencies with AI

```bash
# Create project
mkdir virtual-tryon && cd virtual-tryon

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with AI dependencies
pip install -r requirements.txt
```

#### 5. Run with Real AI

```bash
python app.py
```

**✅ Result**: App will show "🤖 Real Imagen 3 AI" status and use actual Google Cloud AI!

---

### **Option B: Quick Demo Setup (No AI Setup Required)**

#### 1. Basic Setup

```bash
mkdir virtual-tryon && cd virtual-tryon
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install basic dependencies (skip google-cloud-aiplatform)
pip install Flask==2.3.3 Werkzeug==2.3.7 Pillow==10.0.1 opencv-python==4.8.1.78 numpy==1.24.3
```

#### 2. Run Demo Mode

```bash
python app.py
```

**✅ Result**: App will show "🎭 Enhanced Mock AI" status with advanced simulation!

## 🔧 **Project Structure**

```
virtual-tryon/
├── app.py                 # Main Flask app with Real Imagen 3 integration
├── requirements.txt       # Dependencies (including Google Cloud AI)
├── templates/
│   └── index.html        # UI with AI status indicators
├── results/              # AI-generated results
└── README.md            # This documentation
```

## 🔐 **Security & Configuration**

### **Environment Variables:**

```bash
# Required for Real AI Mode
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Optional Configuration
FLASK_ENV=production
MAX_UPLOAD_SIZE=16777216  # 16MB
```

### **Service Account Permissions:**

Your Google Cloud service account needs:

- `aiplatform.user`
- `ml.developer`
- `storage.objectViewer` (if using Cloud Storage)

## 📊 **API Endpoints**

### **New AI Status Endpoint:**

```bash
GET /api/status
```

Returns:

```json
{
  "imagen3_available": true,
  "project_configured": true,
  "credentials_set": true
}
```

### **Enhanced Processing Endpoint:**

```bash
POST /process
```

Now returns:

```json
{
  "success": true,
  "result_image": "result_filename.jpg",
  "mask_image": "mask_filename.png",
  "ai_used": "Real Imagen 3 API",
  "message": "Virtual try-on completed using Real Imagen 3 API"
}
```

## 🚀 **Production Deployment**

### **Google Cloud Run Deployment:**

```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/virtual-tryon

# Deploy
gcloud run deploy virtual-tryon \
  --image gcr.io/PROJECT_ID/virtual-tryon \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=PROJECT_ID
```

### **Scaling Configuration:**

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: virtual-tryon
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/timeout: "300s"
```
