import spotipy
from spotipy.oauth2 import SpotifyOAuth
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

SPOTIPY_CLIENT_ID = "${{secrets.SPOTIPY_CLIENT_ID}}"
SPOTIPY_CLIENT_SECRET = "${{secrets.SPOTIPY_CLIENT_SECRET}}"
SPOTIPY_REDIRECT_URI = "http://localhost:8888/callback"
SCOPE = "user-modify-playback-state user-read-playback-state"

# Authenticate and create a Spotipy object
try:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SCOPE,
    ))
    logger.info("Spotify authentication successful.")
except Exception as e:
    logger.error(f"Error during Spotify authentication: {e}")
    raise

def get_devices():
    """
    Get the user's available devices.
    """
    logger.info("Fetching available devices.")
    try:
        devices = sp.devices()
        logger.debug(f"Devices Response: {devices}")
        if not devices['devices']:
            logger.warning("No active devices found. Open Spotify on one of your devices.")
            return None
        return devices['devices']
    except Exception as e:
        logger.error(f"Error fetching devices: {e}")
        return None

def play_song(song_uri):
    """
    Play a song using its Spotify URI.
    :param song_uri: The Spotify URI of the song (e.g., spotify:track:2P5cIXejqLpHDQeCHAbbBG).
    """
    logger.info(f"Attempting to play song: {song_uri}")
    devices = get_devices()
    if not devices:
        logger.error("No devices available for playback.")
        return
    device_id = devices[0]['id']

    try:
        sp.start_playback(device_id=device_id, uris=[song_uri])
        logger.info(f"Playing song: {song_uri} on device {device_id}")
    except Exception as e:
        logger.error(f"Error starting playback: {e}")

if __name__ == "__main__":
    song_uri = "spotify:track:2P5cIXejqLpHDQeCHAbbBG"
    play_song(song_uri)
