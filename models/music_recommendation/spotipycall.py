import spotipy
from spotipy.oauth2 import SpotifyOAuth
SPOTIPY_CLIENT_ID = ${{secrets.SPOTIPY_CLIENT_ID}}
SPOTIPY_CLIENT_SECRET = ${{secrets.SPOTIPY_CLIENT_SECRET}}
SPOTIPY_REDIRECT_URI = "http://localhost:8888/callback"
SCOPE = "user-modify-playback-state user-read-playback-state"

# Authenticate and create a Spotipy object
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE,
))

def get_devices():
    """
    Get the user's available devices.
    """
    devices = sp.devices()
    print("Devices Response:", devices) 
    if not devices['devices']:
        print("No active devices found. Open Spotify on one of your devices.")
        return None
    return devices['devices']

def play_song(song_uri):
    """
    Play a song using its Spotify URI.
    :param song_uri: The Spotify URI of the song (e.g., spotify:track:2P5cIXejqLpHDQeCHAbbBG).
    """
    devices = get_devices()
    if not devices:
        return
    device_id = devices[0]['id']

    try:
        sp.start_playback(device_id=device_id, uris=[song_uri])
        print(f"Playing song: {song_uri}")
    except Exception as e:
        print(f"Error starting playback: {e}")

if __name__ == "__main__":
    song_uri = "spotify:track:2P5cIXejqLpHDQeCHAbbBG"
    play_song(song_uri)