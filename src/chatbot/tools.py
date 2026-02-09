"""Tool infrastructure for the chatbot system."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result from executing a tool."""

    success: bool
    output: Any
    error: str | None = None


class ToolCall(BaseModel):
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class Tool(ABC):
    """Abstract base class for tools."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for parameters

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given arguments."""
        pass

    def get_openai_schema(self) -> dict[str, Any]:
        """Get the OpenAI function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class PythonExecuteTool(Tool):
    """Execute Python code in a sandboxed environment."""

    name = "python_execute"
    description = "Execute Python code and return the result. Use for calculations, data processing, and analysis."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute",
            },
        },
        "required": ["code"],
    }

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def execute(self, code: str, **kwargs: Any) -> ToolResult:
        """Execute Python code with restricted globals."""
        try:
            # Capture stdout
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Restricted globals for safety
            restricted_globals = {
                "__builtins__": {
                    "__import__": __import__,
                    "print": print,
                    "len": len,
                    "range": range,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "reversed": reversed,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "any": any,
                    "all": all,
                    "isinstance": isinstance,
                    "type": type,
                    "hasattr": hasattr,
                    "getattr": getattr,
                    "setattr": setattr,
                    "callable": callable,
                    "iter": iter,
                    "next": next,
                    "repr": repr,
                    "hash": hash,
                    "id": id,
                    "dir": dir,
                    "vars": vars,
                    "open": open,
                    "True": True,
                    "False": False,
                    "None": None,
                    "Exception": Exception,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "KeyError": KeyError,
                    "IndexError": IndexError,
                    "AttributeError": AttributeError,
                    "StopIteration": StopIteration,
                },
            }

            def run_code() -> Any:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                    stderr_capture
                ):
                    # Use single dict for globals+locals so imports are
                    # visible inside function bodies
                    exec(code, restricted_globals)
                return restricted_globals.get("result", stdout_capture.getvalue().strip())

            # Run with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_code),
                timeout=self.timeout,
            )

            output = result if result else stdout_capture.getvalue().strip()
            stderr_output = stderr_capture.getvalue().strip()

            if stderr_output:
                return ToolResult(success=True, output=output, error=stderr_output)

            return ToolResult(success=True, output=output or "Code executed successfully (no output)")

        except asyncio.TimeoutError:
            return ToolResult(success=False, output=None, error="Code execution timed out")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WebSearchTool(Tool):
    """Search the web for information."""

    name = "web_search"
    description = "Search the web for current information, facts, and news."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
        self._last_search_time: float = 0

    async def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> ToolResult:
        """Search the web using DuckDuckGo."""
        try:
            import time

            # Rate limiting
            current_time = time.time()
            if current_time - self._last_search_time < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - (current_time - self._last_search_time))
            self._last_search_time = time.time()

            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS
                except ImportError:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="ddgs package not installed. Run: pip install ddgs",
                    )

            loop = asyncio.get_event_loop()

            def search() -> list[dict]:
                ddgs = DDGS()
                results = list(ddgs.text(query, max_results=max_results))
                return results

            results = await loop.run_in_executor(None, search)

            formatted_results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]

            return ToolResult(success=True, output={"results": formatted_results})

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileReadTool(Tool):
    """Read contents of a file."""

    name = "file_read"
    description = "Read the contents of a file from the allowed directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to read (relative to allowed directory)",
            },
        },
        "required": ["path"],
    }

    def __init__(self, allowed_directory: str | Path | None = None):
        self.allowed_directory = Path(allowed_directory) if allowed_directory else Path.cwd()

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve the path, ensuring it's within allowed directory."""
        resolved = (self.allowed_directory / path).resolve()
        if not str(resolved).startswith(str(self.allowed_directory.resolve())):
            raise ValueError(f"Access denied: path '{path}' is outside allowed directory")
        return resolved

    async def execute(self, path: str, **kwargs: Any) -> ToolResult:
        """Read file contents."""
        try:
            resolved_path = self._validate_path(path)

            if not resolved_path.exists():
                return ToolResult(success=False, output=None, error=f"File not found: {path}")

            if not resolved_path.is_file():
                return ToolResult(success=False, output=None, error=f"Not a file: {path}")

            content = resolved_path.read_text()
            return ToolResult(success=True, output={"content": content, "path": str(resolved_path)})

        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileWriteTool(Tool):
    """Write contents to a file."""

    name = "file_write"
    description = "Write content to a file in the allowed directory. Creates parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to write (relative to allowed directory)",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, allowed_directory: str | Path | None = None):
        self.allowed_directory = Path(allowed_directory) if allowed_directory else Path.cwd()

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve the path, ensuring it's within allowed directory."""
        resolved = (self.allowed_directory / path).resolve()
        if not str(resolved).startswith(str(self.allowed_directory.resolve())):
            raise ValueError(f"Access denied: path '{path}' is outside allowed directory")
        return resolved

    async def execute(self, path: str, content: str, **kwargs: Any) -> ToolResult:
        """Write content to file."""
        try:
            resolved_path = self._validate_path(path)

            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            resolved_path.write_text(content)
            return ToolResult(success=True, output={"success": True, "path": str(resolved_path)})

        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileUpdateTool(Tool):
    """Update an existing file by appending content or replacing text."""

    name = "file_update"
    description = (
        "Update an existing file. Use mode 'append' to add content at the end, "
        "or mode 'replace' to find and replace specific text within the file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file to update (relative to allowed directory)",
            },
            "content": {
                "type": "string",
                "description": "The content to append, or the replacement text (when using replace mode)",
            },
            "mode": {
                "type": "string",
                "enum": ["append", "replace"],
                "description": "Update mode: 'append' to add at end, 'replace' to find and replace",
                "default": "append",
            },
            "find": {
                "type": "string",
                "description": "The text to find (required when mode is 'replace')",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, allowed_directory: str | Path | None = None):
        self.allowed_directory = Path(allowed_directory) if allowed_directory else Path.cwd()

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve the path, ensuring it's within allowed directory."""
        resolved = (self.allowed_directory / path).resolve()
        if not str(resolved).startswith(str(self.allowed_directory.resolve())):
            raise ValueError(f"Access denied: path '{path}' is outside allowed directory")
        return resolved

    async def execute(
        self, path: str, content: str, mode: str = "append", find: str = "", **kwargs: Any
    ) -> ToolResult:
        """Update a file by appending or replacing content."""
        try:
            resolved_path = self._validate_path(path)

            if not resolved_path.exists():
                return ToolResult(success=False, output=None, error=f"File not found: {path}")

            if not resolved_path.is_file():
                return ToolResult(success=False, output=None, error=f"Not a file: {path}")

            existing = resolved_path.read_text()

            if mode == "append":
                resolved_path.write_text(existing + content)
                return ToolResult(
                    success=True,
                    output={"path": str(resolved_path), "mode": "append", "bytes_added": len(content)},
                )
            elif mode == "replace":
                if not find:
                    return ToolResult(
                        success=False, output=None, error="'find' parameter is required for replace mode"
                    )
                if find not in existing:
                    return ToolResult(
                        success=False, output=None, error=f"Text to find not present in file: {path}"
                    )
                updated = existing.replace(find, content)
                resolved_path.write_text(updated)
                count = existing.count(find)
                return ToolResult(
                    success=True,
                    output={"path": str(resolved_path), "mode": "replace", "replacements": count},
                )
            else:
                return ToolResult(
                    success=False, output=None, error=f"Unknown mode: {mode}. Use 'append' or 'replace'."
                )

        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileDeleteTool(Tool):
    """Delete a file or empty directory."""

    name = "file_delete"
    description = "Delete a file or an empty directory from the allowed directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path to the file or empty directory to delete (relative to allowed directory)",
            },
            "recursive": {
                "type": "boolean",
                "description": "If true, recursively delete a directory and all its contents. Use with caution.",
                "default": False,
            },
        },
        "required": ["path"],
    }

    def __init__(self, allowed_directory: str | Path | None = None):
        self.allowed_directory = Path(allowed_directory) if allowed_directory else Path.cwd()

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve the path, ensuring it's within allowed directory."""
        resolved = (self.allowed_directory / path).resolve()
        allowed_root = self.allowed_directory.resolve()
        if not str(resolved).startswith(str(allowed_root)):
            raise ValueError(f"Access denied: path '{path}' is outside allowed directory")
        if resolved == allowed_root:
            raise ValueError("Access denied: cannot delete the allowed root directory")
        return resolved

    async def execute(self, path: str, recursive: bool = False, **kwargs: Any) -> ToolResult:
        """Delete a file or directory."""
        try:
            import shutil

            resolved_path = self._validate_path(path)

            if not resolved_path.exists():
                return ToolResult(success=False, output=None, error=f"Path not found: {path}")

            if resolved_path.is_file():
                resolved_path.unlink()
                return ToolResult(
                    success=True, output={"deleted": str(resolved_path), "type": "file"}
                )
            elif resolved_path.is_dir():
                if recursive:
                    shutil.rmtree(resolved_path)
                    return ToolResult(
                        success=True,
                        output={"deleted": str(resolved_path), "type": "directory", "recursive": True},
                    )
                else:
                    try:
                        resolved_path.rmdir()
                    except OSError:
                        return ToolResult(
                            success=False,
                            output=None,
                            error=f"Directory not empty: {path}. Set recursive=true to delete with contents.",
                        )
                    return ToolResult(
                        success=True, output={"deleted": str(resolved_path), "type": "directory"}
                    )
            else:
                return ToolResult(success=False, output=None, error=f"Unsupported path type: {path}")

        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class DirectoryListTool(Tool):
    """List contents of a directory."""

    name = "directory_list"
    description = "List files and subdirectories in a directory. Returns names, types, and sizes."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The directory path to list (relative to allowed directory). Use '.' for the root.",
                "default": ".",
            },
        },
        "required": [],
    }

    def __init__(self, allowed_directory: str | Path | None = None):
        self.allowed_directory = Path(allowed_directory) if allowed_directory else Path.cwd()

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve the path, ensuring it's within allowed directory."""
        resolved = (self.allowed_directory / path).resolve()
        if not str(resolved).startswith(str(self.allowed_directory.resolve())):
            raise ValueError(f"Access denied: path '{path}' is outside allowed directory")
        return resolved

    async def execute(self, path: str = ".", **kwargs: Any) -> ToolResult:
        """List directory contents."""
        try:
            resolved_path = self._validate_path(path)

            if not resolved_path.exists():
                return ToolResult(success=False, output=None, error=f"Directory not found: {path}")

            if not resolved_path.is_dir():
                return ToolResult(success=False, output=None, error=f"Not a directory: {path}")

            entries = []
            for entry in sorted(resolved_path.iterdir()):
                info = {
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                }
                if entry.is_file():
                    info["size"] = entry.stat().st_size
                entries.append(info)

            return ToolResult(
                success=True,
                output={"path": str(resolved_path), "entries": entries, "count": len(entries)},
            )

        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class DirectoryCreateTool(Tool):
    """Create a directory."""

    name = "directory_create"
    description = "Create a new directory (including parent directories if needed)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path of the directory to create (relative to allowed directory)",
            },
        },
        "required": ["path"],
    }

    def __init__(self, allowed_directory: str | Path | None = None):
        self.allowed_directory = Path(allowed_directory) if allowed_directory else Path.cwd()

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve the path, ensuring it's within allowed directory."""
        resolved = (self.allowed_directory / path).resolve()
        if not str(resolved).startswith(str(self.allowed_directory.resolve())):
            raise ValueError(f"Access denied: path '{path}' is outside allowed directory")
        return resolved

    async def execute(self, path: str, **kwargs: Any) -> ToolResult:
        """Create a directory."""
        try:
            resolved_path = self._validate_path(path)

            if resolved_path.exists():
                if resolved_path.is_dir():
                    return ToolResult(
                        success=True,
                        output={"path": str(resolved_path), "already_existed": True},
                    )
                return ToolResult(
                    success=False, output=None, error=f"A file already exists at: {path}"
                )

            resolved_path.mkdir(parents=True, exist_ok=True)
            return ToolResult(
                success=True,
                output={"path": str(resolved_path), "created": True},
            )

        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WebDownloadTool(Tool):
    """Download content from a URL."""

    name = "web_download"
    description = (
        "Download content from a URL. Returns the text content of the page. "
        "Optionally save to a file. Useful for fetching web pages, APIs, or raw data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to download content from",
            },
            "save_path": {
                "type": "string",
                "description": "Optional file path to save the downloaded content (relative to allowed directory)",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum number of characters to return (default: 50000)",
                "default": 50000,
            },
        },
        "required": ["url"],
    }

    def __init__(self, allowed_directory: str | Path | None = None, timeout: float = 30.0):
        self.allowed_directory = Path(allowed_directory) if allowed_directory else Path.cwd()
        self.timeout = timeout

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve the path, ensuring it's within allowed directory."""
        resolved = (self.allowed_directory / path).resolve()
        if not str(resolved).startswith(str(self.allowed_directory.resolve())):
            raise ValueError(f"Access denied: path '{path}' is outside allowed directory")
        return resolved

    async def execute(self, url: str, save_path: str = "", max_length: int = 50000, **kwargs: Any) -> ToolResult:
        """Download content from a URL."""
        try:
            import urllib.request
            import urllib.error
            from html.parser import HTMLParser

            loop = asyncio.get_event_loop()

            def download() -> tuple[str, int]:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    charset = resp.headers.get_content_charset() or "utf-8"
                    content = raw.decode(charset, errors="replace")
                    return content, len(raw)

            content, raw_size = await asyncio.wait_for(
                loop.run_in_executor(None, download),
                timeout=self.timeout + 5,
            )

            # Simple HTML tag stripping for readability
            class TagStripper(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.result = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "noscript"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "noscript"):
                        self._skip = False
                    if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li", "tr"):
                        self.result.append("\n")

                def handle_data(self, data):
                    if not self._skip:
                        self.result.append(data)

            stripper = TagStripper()
            stripper.feed(content)
            text = "".join(stripper.result).strip()

            # Truncate if needed
            truncated = False
            if len(text) > max_length:
                text = text[:max_length]
                truncated = True

            # Save to file if requested
            saved_path = None
            if save_path:
                resolved = self._validate_path(save_path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(text)
                saved_path = str(resolved)

            return ToolResult(
                success=True,
                output={
                    "content": text,
                    "url": url,
                    "raw_size": raw_size,
                    "text_length": len(text),
                    "truncated": truncated,
                    "saved_to": saved_path,
                },
            )

        except asyncio.TimeoutError:
            return ToolResult(success=False, output=None, error=f"Download timed out for: {url}")
        except urllib.error.HTTPError as e:
            return ToolResult(success=False, output=None, error=f"HTTP error {e.code}: {e.reason} for {url}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_openai_schema(self, allowed_tools: list[str] | None = None) -> list[dict[str, Any]]:
        """Get OpenAI function schemas for all registered tools."""
        tools = self._tools.values()
        if allowed_tools is not None:
            tools = [t for t in tools if t.name in allowed_tools]
        return [tool.get_openai_schema() for tool in tools]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, output=None, error=f"Tool not found: {name}")

        return await tool.execute(**arguments)


def create_default_registry(
    allowed_directory: str | Path | None = None,
    python_timeout: float = 30.0,
    web_search_rate_limit: float = 1.0,
) -> ToolRegistry:
    """Create a registry with all default tools."""
    registry = ToolRegistry()

    registry.register(PythonExecuteTool(timeout=python_timeout))
    registry.register(WebSearchTool(rate_limit_delay=web_search_rate_limit))
    registry.register(FileReadTool(allowed_directory=allowed_directory))
    registry.register(FileWriteTool(allowed_directory=allowed_directory))
    registry.register(FileUpdateTool(allowed_directory=allowed_directory))
    registry.register(FileDeleteTool(allowed_directory=allowed_directory))
    registry.register(DirectoryListTool(allowed_directory=allowed_directory))
    registry.register(DirectoryCreateTool(allowed_directory=allowed_directory))
    registry.register(WebDownloadTool(allowed_directory=allowed_directory))

    return registry
