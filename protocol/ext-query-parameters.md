(This document describes an extension that front-ends aren't required to implement. Front-ends that don't implement this extension should reject attempts to set the `query-parameters` front-end configuration option.)

Many algorithms expose parameters that can be changed to adjust their search strategies without requiring that training data be resubmitted. When the front-end configuration option `query-parameters` is set to `1`, a new command will be added to query mode allowing these query configuration parameters to be changed.

(Front-ends that support other optional query modes, such as prepared or batch queries, should also add this command to those modes.)

## Commands

### Configuration mode

#### `frontend query-parameters V` (three tokens)

If `V` is `1`, then request that query mode expose the `query-params` command. If `V` is anything else, then withdraw this request.

Responses:

* `epbprtv0 ok`

  The availability of the `query-params` command has been changed accordingly.

* `epbprtv0 fail`

  This command has had no effect on the availability of the `query-params` command.

### Training mode

This extension makes no changes to training mode.

### Query mode

When the `query-parameters` front-end configuration option has been set to `1`, this extension adds one new command to query mode:

#### `query-params [VALUE0, ..., VALUEk] set` (two or more tokens)

Change the values of the query parameters.

(The final token `set` is required. It exists for the sake of compatibility with the `batch-queries` extension, which also uses variable-length commands but which requires that the last token specify a number.)

Responses:

* `epbprtv0 ok`

  The query parameters were changed to the given values.

* `epbprtv0 fail`

  The query parameters were not changed to the given values, perhaps because one of them was invalid.
