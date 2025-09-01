import requests

def send_to_n8n(payload:dict, n8n_webhook_url:str):
    try:
        response = requests.post(n8n_webhook_url, json=payload,timeout=5)
        response.raise_for_status()
        print(f"Successfully sent payload to n8n: {payload}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending payload to n8n: {e}")