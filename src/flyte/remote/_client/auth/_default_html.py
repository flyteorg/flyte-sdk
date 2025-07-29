import textwrap


def get_default_success_html(endpoint: str) -> str:
    """Get default success html."""
    return textwrap.dedent(
        """
    <html>
    <head>
        <title>OAuth2 Authentication to Union Successful</title>
    </head>
    <body style="background:white;font-family:Arial">
        <div style="position: absolute;top:40%;left:50%;transform: translate(-50%, -50%);text-align:center;">
            <div style="margin:auto">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 65" fill="currentColor"
                    style="color:#fdb51e;width:360px;">
                    <title>Union.ai</title>
                    <path d="M32,64.8C14.4,64.8,0,51.5,0,34V3.6h17.6v41.3c0,1.9,1.1,3,3,3h23c1.9,0,3-1.1,3-3V3.6H64V34
                    C64,51.5,49.6,64.8,32,64.8z M69.9,30.9v30.4h17.6V20c0-1.9,1.1-3,3-3h23c1.9,0,3,1.1,3,3v41.3H134V30.9c0-17.5-14.4-30.8-32.1-30.8
                    S69.9,13.5,69.9,30.9z M236,30.9v30.4h17.6V20c0-1.9,1.1-3,3-3h23c1.9,0,3,1.1,3,3v41.3H300V30.9c0-17.5-14.4-30.8-32-30.8
                    S236,13.5,236,30.9L236,30.9z M230.1,32.4c0,18.2-14.2,32.5-32.2,32.5s-32-14.3-32-32.5s14-32.1,32-32.1S230.1,14.3,230.1,32.4
                    L230.1,32.4z M213.5,20.2c0-1.9-1.1-3-3-3h-24.8c-1.9,0-3,1.1-3,3v24.5c0,1.9,1.1,3,3,3h24.8c1.9,0,3-1.1,3-3V20.2z M158.9,3.6
                    h-17.6v57.8h17.6V3.6z"></path>
                </svg>
                <h2>You've successfully authenticated to Union!</h2>
                <p style="font-size:20px;">Return to your terminal for next steps</p>
            </div>
        </div>
    </body>
    </html>
    """  # noqa: E501
    )
