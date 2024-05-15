import dartmouth_auth


class AuthenticatedMixin:
    """A mix-in class to faciliate authentication"""

    def authenticate(self, jwt_url=None):
        if self.authenticator:
            jwt = self.authenticator()
        else:
            jwt = dartmouth_auth.get_jwt(
                dartmouth_api_key=self.dartmouth_api_key, jwt_url=jwt_url
            )
        auth_header = {"Authorization": f"Bearer {jwt}"}
        if self.client.headers is not None:
            self.client.headers.update(auth_header)
        else:
            self.client.headers = auth_header
        if hasattr(self, "async_client"):
            if self.async_client.headers is not None:
                self.async_client.headers.update(auth_header)
            else:
                self.async_client.headers = auth_header
