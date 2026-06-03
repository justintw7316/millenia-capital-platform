"""
integrations/apollo_client.py — Apollo.io contact enrichment.

https://apolloio.github.io/apollo-api-docs/
Used in Step 7a for investor contact enrichment.
"""
import requests
from typing import List, Dict

from config import APOLLO_API_KEY
from core.logger import get_logger

logger = get_logger(__name__)

APOLLO_BASE = "https://api.apollo.io/v1"


class ApolloClient:
    """Apollo.io client for investor search and contact enrichment."""

    def __init__(self):
        self._headers = {
            "X-Api-Key": APOLLO_API_KEY,
            "Content-Type": "application/json",
        }

    def search_investors(self, criteria: Dict) -> List[Dict]:
        """
        Search for investors matching given criteria.

        Args:
            criteria: {industry, title_keywords, location, min_portfolio_size}
        Returns:
            List of investor contact dicts.
        """
        try:
            body = {
                "q_keywords": criteria.get("industry", ""),
                "person_titles": criteria.get("title_keywords", []),
                "person_locations": criteria.get("location", []),
                "page": 1,
                "per_page": 25,
            }
            resp = requests.post(
                f"{APOLLO_BASE}/mixed_people/search",
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()
            people = resp.json().get("people", [])
            logger.info(f"[Apollo] search_investors returned {len(people)} contacts")
            return people
        except Exception as e:
            logger.error(f"[Apollo] search_investors failed: {e}")
            return []

    def enrich_contact(self, name: str, company: str) -> Dict:
        """
        Enrich a contact record with full details.

        Args:
            name: Contact full name.
            company: Company/firm name.
        Returns:
            Dict with person details from Apollo.
        """
        try:
            body = {
                "name": name,
                "organization_name": company,
                "reveal_personal_emails": True,
            }
            resp = requests.post(
                f"{APOLLO_BASE}/people/match",
                headers=self._headers,
                json=body,
            )
            resp.raise_for_status()
            person = resp.json().get("person", {})
            logger.info(f"[Apollo] Enriched {name!r} → {person.get('email', 'N/A')}")
            return person
        except Exception as e:
            logger.error(f"[Apollo] enrich_contact failed for {name!r}: {e}")
            return {}
