import unittest
from unittest import mock

import grpc
import grpc.aio
from flyteidl.admin import project_pb2

from flyte.remote._client.controlplane import ClientSet


class TestDnsEndpointWithPkce(unittest.TestCase):
    """Test cases for the for_endpoint function with DNS URI and PKCE auth."""

    @mock.patch("flyte.remote.client.controlplane.create_channel")
    async def test_for_endpoint_with_dns_and_pkce(self, mock_create_channel):
        """Test creating a client with DNS URI endpoint and PKCE authentication."""
        # Setup mocks
        mock_channel = mock.AsyncMock(spec=grpc.aio.Channel)
        mock_create_channel.return_value = mock_channel

        # Mock the admin client and its ListProjects method
        mock_admin_client = mock.AsyncMock()
        mock_projects_response = project_pb2.Projects()
        mock_admin_client.ListProjects.return_value = mock_projects_response

        # Create a ClientSet with mocked internals
        with mock.patch.object(ClientSet, "_admin_client", mock_admin_client):
            # Call the function with DNS URI and PKCE auth
            endpoint = "dns:///dogfood.cloud-staging.flyte.ai"
            client_set = ClientSet.for_endpoint(
                endpoint=endpoint,
                insecure=False,  # Secure connection
                auth_type="Pkce",  # Use PKCE authentication
            )

            # Verify create_channel was called with correct parameters
            mock_create_channel.assert_called_once_with(endpoint=endpoint, insecure=False, auth_type="Pkce")

            # Call ListProjects and verify it works
            request = project_pb2.ProjectListRequest()
            await client_set.project_domain_service.ListProjects(request)

            # Verify ListProjects was called
            mock_admin_client.ListProjects.assert_called_once_with(request)
