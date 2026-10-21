"""Configuración de logging para los comandos de consola (con UTF-8 en Windows)."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Mensajes limpios en consola; fuerza UTF-8 para que las tildes se vean bien en Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=level, format="%(message)s")
