"""Read-only Twitch chat reader over IRC.

Twitch chat is plain IRC. We connect anonymously (an unauthenticated
``justinfan`` login can read any public channel — no OAuth, no token, no app
registration), join one channel, and buffer incoming messages on a background
thread. Call :meth:`TwitchChat.poll` to drain everything received since the
last call.

Read-only by design: we never send chat messages, which is exactly why the
anonymous login is enough. If posting to chat or guaranteed delivery is ever
needed, that's the point to move to Twitch EventSub (which requires an OAuth
app).

Standalone smoke test against any live public channel (no account needed):

    python -m pokemon_agent.chat.twitch <channel>
"""

import logging
import random
import socket
import threading
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667  # plain TCP; read-only public chat needs no TLS


@dataclass(frozen=True)
class ChatMessage:
    author: str
    text: str


def parse_irc_message(line: str) -> ChatMessage | None:
    """Parse one raw IRC line into a ChatMessage, or None if it isn't chat.

    Handles the IRCv3-tagged and untagged forms Twitch sends:

        @...;display-name=Alice;... :alice!alice@alice.tmi.twitch.tv PRIVMSG #chan :hi
        :alice!alice@alice.tmi.twitch.tv PRIVMSG #chan :hi
    """
    rest = line
    tags: dict[str, str] = {}
    if rest.startswith("@"):
        tag_part, _, rest = rest[1:].partition(" ")
        for item in tag_part.split(";"):
            key, _, val = item.partition("=")
            tags[key] = val

    prefix = ""
    if rest.startswith(":"):
        prefix, _, rest = rest[1:].partition(" ")

    # rest is now "COMMAND params... :trailing". The first " :" is the trailing
    # delimiter, so message text keeps any later colons (e.g. emoticons).
    command_part, sep, trailing = rest.partition(" :")
    fields = command_part.split()
    if not fields or fields[0] != "PRIVMSG" or not sep:
        return None

    nick = prefix.split("!", 1)[0]
    author = tags.get("display-name") or nick or "unknown"
    return ChatMessage(author=author, text=trailing)


class TwitchChat:
    """Background reader of one Twitch channel's chat (anonymous, read-only)."""

    CONNECT_TIMEOUT = 10.0
    RECV_TIMEOUT = 1.0       # so the read loop can notice stop() promptly
    MAX_BACKOFF = 30.0

    def __init__(
        self,
        channel: str,
        *,
        host: str = IRC_HOST,
        port: int = IRC_PORT,
        buffer_size: int = 200,
    ):
        self.channel = channel.lstrip("#").lower()
        self.host = host
        self.port = port
        self._buffer: deque[ChatMessage] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._nick = f"justinfan{random.randint(10000, 99999)}"

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="twitch-chat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()  # unblock a recv() in progress
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def poll(self) -> list[ChatMessage]:
        """Return and clear every message received since the last poll."""
        with self._lock:
            msgs = list(self._buffer)
            self._buffer.clear()
        return msgs

    # -- internals ---------------------------------------------------------- #

    def _append(self, msg: ChatMessage) -> None:
        with self._lock:
            self._buffer.append(msg)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._connect_and_read()
                backoff = 1.0
            except OSError as e:
                if self._stop.is_set():
                    break
                logger.warning(
                    "[Twitch] connection error (%s); reconnecting in %.0fs", e, backoff
                )
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.MAX_BACKOFF)

    def _connect_and_read(self) -> None:
        sock = socket.create_connection(
            (self.host, self.port), timeout=self.CONNECT_TIMEOUT
        )
        self._sock = sock
        sock.settimeout(self.RECV_TIMEOUT)
        self._handshake(sock)
        logger.info("[Twitch] connected, joined #%s as %s", self.channel, self._nick)

        recv_buffer = ""
        while not self._stop.is_set():
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            if not data:
                raise OSError("connection closed by server")
            recv_buffer += data.decode("utf-8", errors="replace")
            *lines, recv_buffer = recv_buffer.split("\r\n")  # keep partial tail
            for line in lines:
                self._handle_line(sock, line)

        try:
            sock.close()
        except OSError:
            pass

    def _handshake(self, sock: socket.socket) -> None:
        # Tags give us display-name; anonymous login needs NICK only (no PASS).
        sock.sendall(b"CAP REQ :twitch.tv/tags twitch.tv/commands\r\n")
        sock.sendall(f"NICK {self._nick}\r\n".encode())
        sock.sendall(f"JOIN #{self.channel}\r\n".encode())

    def _handle_line(self, sock: socket.socket, line: str) -> None:
        if not line:
            return
        if line.startswith("PING"):
            token = line[4:].strip()
            try:
                sock.sendall(f"PONG {token}\r\n".encode())
            except OSError:
                pass
            return
        msg = parse_irc_message(line)
        if msg is not None:
            self._append(msg)


def _main(argv: list[str]) -> int:
    import time

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) < 2:
        print("usage: python -m pokemon_agent.chat.twitch <channel>")
        return 2

    chat = TwitchChat(argv[1])
    chat.start()
    print(f"Reading #{chat.channel} chat (Ctrl-C to stop)...")
    try:
        while True:
            for m in chat.poll():
                print(f"{m.author}: {m.text}", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        chat.stop()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv))
