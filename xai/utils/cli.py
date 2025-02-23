import click


def path(func):
    return click.option(
        "--path",
        default="params.yaml",
        help="Path to configuration yaml file",
        type=str,
    )(func)
