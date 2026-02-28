# spotify_player.py - Spotify API (CORREGIDO)
# Requiere cuenta Spotify Premium para control de reproducción
import os
import time
from typing import Optional, TYPE_CHECKING, Any
from dotenv import load_dotenv

load_dotenv()

# Type checking imports
if TYPE_CHECKING:
    import spotipy as spotipy_module
    from spotipy.oauth2 import SpotifyOAuth as SpotifyOAuthClass
else:
    spotipy_module = None  # type: ignore
    SpotifyOAuthClass = None  # type: ignore

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    spotipy = None  # type: ignore
    SpotifyOAuth = None  # type: ignore
    print("⚠️  spotipy no instalado: pip install spotipy")


# Scopes necesarios para controlar reproducción
SPOTIFY_SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "streaming",
    "playlist-read-private",
])


class SpotifyPlayer:
    def __init__(self):
        self._client: Optional[Any] = None  # spotipy.Spotify
        self._active_device_id: Optional[str] = None

    @property
    def client(self) -> Optional[Any]:  # spotipy.Spotify
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_client(self) -> Optional[Any]:  # spotipy.Spotify
        if not SPOTIPY_AVAILABLE or spotipy is None or SpotifyOAuth is None:
            return None

        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

        if not client_id or not client_secret:
            print("❌ SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET no encontrados en .env")
            print("   Obtené tus credenciales en: https://developer.spotify.com/dashboard")
            return None

        try:
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=SPOTIFY_SCOPES,
                cache_path=".spotify_cache",
                open_browser=True  # Abre el navegador para auth la primera vez
            )
            sp = spotipy.Spotify(auth_manager=auth_manager)
            # Test de conexión
            sp.current_user()
            print("✅ Spotify conectado")
            return sp
        except Exception as e:
            print(f"❌ Error conectando Spotify: {e}")
            return None

    def is_available(self) -> bool:
        return SPOTIPY_AVAILABLE and self.client is not None

    def _get_active_device(self) -> Optional[str]:
        """Obtiene el ID del dispositivo activo"""
        if not self.client:
            return None
        try:
            devices = self.client.devices()
            device_list = devices.get("devices", []) if devices else []

            if not device_list:
                print("⚠️  No hay dispositivos Spotify activos")
                print("   Abrí Spotify en tu PC, teléfono o navegador primero")
                return None

            # Preferir dispositivo activo, sino el primero
            active = next((d for d in device_list if d["is_active"]), device_list[0])
            self._active_device_id = active["id"]
            print(f"🎵 Dispositivo: {active['name']}")
            return self._active_device_id

        except Exception as e:
            print(f"❌ Error obteniendo dispositivos: {e}")
            return None

    def play(self, query: str) -> str:
        """Busca y reproduce una canción/artista/playlist"""
        if not self.is_available() or not self.client:
            return "Spotify no está disponible."

        device_id = self._get_active_device()
        if not device_id:
            return "No hay dispositivos Spotify activos. Abrí Spotify primero."

        try:
            # Buscar la canción
            results = self.client.search(q=query, type="track", limit=1)
            tracks = results.get("tracks", {}).get("items", []) if results else []

            if not tracks:
                # Intentar buscar como artista
                results = self.client.search(q=query, type="artist", limit=1)
                artists = results.get("artists", {}).get("items", []) if results else []
                if artists:
                    # Reproducir top tracks del artista
                    artist_id = artists[0]["id"]
                    top_tracks = self.client.artist_top_tracks(artist_id, country="AR")
                    track_uris = [t["uri"] for t in top_tracks.get("tracks", [])[:5]]
                    if track_uris:
                        self.client.start_playback(
                            device_id=device_id,
                            uris=track_uris
                        )
                        artist_name = artists[0]["name"]
                        return f"▶️ {artist_name}"
                return f"No encontré '{query}' en Spotify."

            track = tracks[0]
            track_name = track["name"]
            artist_name = track["artists"][0]["name"]
            track_uri = track["uri"]

            self.client.start_playback(
                device_id=device_id,
                uris=[track_uri]
            )
            return f"▶️ {track_name} — {artist_name}"

        except Exception as e:
            if spotipy and hasattr(spotipy, 'exceptions'):
                if isinstance(e, spotipy.exceptions.SpotifyException):
                    if "Premium" in str(e) or "403" in str(e):
                        return "Se necesita Spotify Premium para controlar la reproducción."
            print(f"❌ Spotify error: {e}")
            return "Error al reproducir en Spotify."

    def pause(self) -> str:
        if not self.is_available() or not self.client:
            return "Spotify no disponible."
        try:
            playback = self.client.current_playback()
            if playback and playback.get("is_playing"):
                self.client.pause_playback()
                return "⏸ Pausado"
            return "No hay nada reproduciéndose."
        except Exception as e:
            print(f"❌ {e}")
            return "No pude pausar."

    def resume(self) -> str:
        if not self.is_available() or not self.client:
            return "Spotify no disponible."
        try:
            device_id = self._get_active_device()
            self.client.start_playback(device_id=device_id)
            return "▶️ Reanudado"
        except Exception as e:
            print(f"❌ {e}")
            return "No pude reanudar."

    def stop(self) -> str:
        return self.pause()  # Spotify no tiene "stop", pause es equivalente

    def volume_up(self) -> str:
        return self._change_volume(+25)

    def volume_down(self) -> str:
        return self._change_volume(-25)

    def _change_volume(self, delta: int) -> str:
        if not self.is_available() or not self.client:
            return "Spotify no disponible."
        try:
            playback = self.client.current_playback()
            if not playback:
                return "No hay reproducción activa."
            current_vol = playback.get("device", {}).get("volume_percent", 50)
            new_vol = max(0, min(100, current_vol + delta))
            device_id = self._get_active_device()
            self.client.volume(new_vol, device_id=device_id)
            icon = "🔊" if delta > 0 else "🔉"
            return f"{icon} {new_vol}%"
        except Exception as e:
            print(f"❌ {e}")
            return "No pude cambiar el volumen."

    def current_track(self) -> str:
        if not self.is_available() or not self.client:
            return "Spotify no disponible."
        try:
            playback = self.client.current_playback()
            if not playback or not playback.get("item"):
                return "No hay nada reproduciéndose."
            track = playback["item"]
            name = track["name"]
            artist = track["artists"][0]["name"]
            return f"🎵 {name} — {artist}"
        except Exception:
            return "No pude obtener la canción actual."
        
    def next_track(self) -> str:
        if not self.is_available() or not self.client:
            return "Spotify no disponible."
        try:
            device_id = self._get_active_device()
            self.client.next_track(device_id=device_id)
            return "⏭ Siguiente"
        except Exception as e:
            print(f"❌ {e}")
            return "No pude pasar a la siguiente."

    def previous_track(self) -> str:
        if not self.is_available() or not self.client:
            return "Spotify no disponible."
        try:
            device_id = self._get_active_device()
            self.client.previous_track(device_id=device_id)
            return "⏮ Anterior"
        except Exception as e:
            print(f"❌ {e}")
            return "No pude volver a la anterior."

# Singleton global
spotify_player = SpotifyPlayer()