# YouTube Data Collection Setup Guide

## Prerequisites
- Python 3.8+
- Google Cloud Platform account
- YouTube Data API v3 enabled

## Setup Steps

### 1. Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing one
3. Enable YouTube Data API v3 for your project

### 2. Create API Credentials
1. Go to Credentials page in Google Cloud Console
2. Click "Create Credentials" → "API Key"
3. Copy your API key
4. (Optional) Restrict the API key to YouTube Data API v3

### 3. Configure API Key
1. Create a `.env` file in project root:
```bash
YOUTUBE_API_KEY=your_api_key_here
```

2. Or set environment variable:
```bash
export YOUTUBE_API_KEY=your_api_key_here
```

### 4. Usage Limits
- Default quota: 10,000 units per day
- Search request cost: 100 units
- Video details request: 1 unit
- Monitor usage in Google Cloud Console

### Troubleshooting
1. **Quota Exceeded Error**
   - Check daily quota usage
   - Request quota increase if needed

2. **Invalid API Key**
   - Verify key is correctly copied
   - Check API restrictions

3. **Rate Limit Error**
   - Implement exponential backoff
   - Reduce request frequency 