from textwrap import dedent

diff_desc_system_prompt = dedent(
    """
    Please generate one or two sentence discriotion for input diff code.
    Input diff code contains two parts: ADD CODE and REMOVE CODE.
    The description should include a clear and concise description of the code change.
    """
)

diff_desc_user_prompt = dedent(
    """
    code diff:

    --- ADD CODE ---
    {add_code}

    --- REMOVE CODE ---
    {remove_code}
    """
)
