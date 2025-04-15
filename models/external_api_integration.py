import requests

def get_external_prediction(api_url: str, params: dict) -> dict:
    """
    Fetch prediction data from an external API.
    :param api_url: URL of the external prediction API
    :param params: Dictionary of query parameters to send with the request
    :return: Parsed JSON response as a dictionary
    """
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from external API: {e}")
        return {}
