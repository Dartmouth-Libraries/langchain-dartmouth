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
        self.client.headers = {"Authorization": f"Bearer {jwt}"}
        self.async_client.headers = {"Authorization": f"Bearer {jwt}"}
