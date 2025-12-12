from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SettingsStore:
    """File-based datastore for hierarchical settings.

    Stores settings in a JSON file with the following structure:
    {
        "ROOT": {"key": value, ...},
        "DOMAIN": {
            "domain_name": {"key": value, ...}
        },
        "PROJECT": {
            "domain_name": {
                "project_name": {"key": value, ...}
            }
        }
    }
    """

    def __init__(self, store_path: Path | None = None):
        """Initialize the settings store.

        Args:
            store_path: Path to the JSON file. If None, uses ~/.flyte/settings.json
        """
        if store_path is None:
            store_path = Path.home() / ".flyte" / "settings.json"

        self.store_path = Path(store_path)
        self._ensure_store_exists()

    def _ensure_store_exists(self) -> None:
        """Create the store file and directory if they don't exist."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write_store({"ROOT": {}, "DOMAIN": {}, "PROJECT": {}})

    def _read_store(self) -> dict[str, Any]:
        """Read the entire store from disk."""
        try:
            with open(self.store_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or missing, reset it
            default_store: dict = {"ROOT": {}, "DOMAIN": {}, "PROJECT": {}}
            self._write_store(default_store)
            return default_store

    def _write_store(self, data: dict[str, Any]) -> None:
        """Write the entire store to disk."""
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_local_settings(self, domain: str | None = None, project: str | None = None) -> dict[str, Any]:
        """Get local settings for a specific scope.

        Args:
            domain: Domain name (optional)
            project: Project name (requires domain)

        Returns:
            Dictionary of local settings for the specified scope
        """
        store = self._read_store()

        if project and domain:
            # PROJECT scope
            return store.get("PROJECT", {}).get(domain, {}).get(project, {})
        elif domain:
            # DOMAIN scope
            return store.get("DOMAIN", {}).get(domain, {})
        else:
            # ROOT scope
            return store.get("ROOT", {})

    def get_effective_settings(
        self, domain: str | None = None, project: str | None = None
    ) -> dict[str, tuple[Any, str, str | None, str | None]]:
        """Get effective settings with inheritance and origin tracking.

        Returns a dictionary where each value is a tuple of:
        (value, scope_type, origin_domain, origin_project)

        Args:
            domain: Domain name (optional)
            project: Project name (requires domain)

        Returns:
            Dictionary mapping setting keys to (value, scope_type, domain, project) tuples
        """
        store = self._read_store()
        effective = {}

        # Start with ROOT settings
        for key, value in store.get("ROOT", {}).items():
            effective[key] = (value, "ROOT", None, None)

        # Override with DOMAIN settings if applicable
        if domain:
            domain_settings = store.get("DOMAIN", {}).get(domain, {})
            for key, value in domain_settings.items():
                effective[key] = (value, "DOMAIN", domain, None)

        # Override with PROJECT settings if applicable
        if project and domain:
            project_settings = store.get("PROJECT", {}).get(domain, {}).get(project, {})
            for key, value in project_settings.items():
                effective[key] = (value, "PROJECT", domain, project)

        return effective

    def set_local_settings(
        self, settings: dict[str, Any], domain: str | None = None, project: str | None = None
    ) -> None:
        """Set local settings for a specific scope, replacing all existing settings.

        Args:
            settings: Dictionary of settings to store
            domain: Domain name (optional)
            project: Project name (requires domain)
        """
        store = self._read_store()

        if project and domain:
            # PROJECT scope
            if "PROJECT" not in store:
                store["PROJECT"] = {}
            if domain not in store["PROJECT"]:
                store["PROJECT"][domain] = {}
            store["PROJECT"][domain][project] = settings
        elif domain:
            # DOMAIN scope
            if "DOMAIN" not in store:
                store["DOMAIN"] = {}
            store["DOMAIN"][domain] = settings
        else:
            # ROOT scope
            store["ROOT"] = settings

        self._write_store(store)

    def delete_scope(self, domain: str | None = None, project: str | None = None) -> None:
        """Delete all settings for a specific scope.

        Args:
            domain: Domain name (optional)
            project: Project name (requires domain)
        """
        store = self._read_store()

        if project and domain:
            # Delete PROJECT scope
            if domain in store.get("PROJECT", {}) and project in store["PROJECT"][domain]:
                del store["PROJECT"][domain][project]
                # Clean up empty domain
                if not store["PROJECT"][domain]:
                    del store["PROJECT"][domain]
        elif domain:
            # Delete DOMAIN scope
            if domain in store.get("DOMAIN", {}):
                del store["DOMAIN"][domain]
            # Also delete all projects in this domain
            if domain in store.get("PROJECT", {}):
                del store["PROJECT"][domain]
        else:
            # Delete ROOT scope (reset to empty)
            store["ROOT"] = {}

        self._write_store(store)

    def list_domains(self) -> list[str]:
        """List all domains that have settings defined.

        Returns:
            List of domain names
        """
        store = self._read_store()
        domains = set()
        domains.update(store.get("DOMAIN", {}).keys())
        domains.update(store.get("PROJECT", {}).keys())
        return sorted(domains)

    def list_projects(self, domain: str) -> list[str]:
        """List all projects in a domain that have settings defined.

        Args:
            domain: Domain name

        Returns:
            List of project names
        """
        store = self._read_store()
        return sorted(store.get("PROJECT", {}).get(domain, {}).keys())
