(This document describes an extension that front-ends aren't required to implement. Front-ends that don't implement this extension should reject attempts to set the `batch-queries` front-end configuration option.)

When the front-end configuration option `batch-queries` is set to `1`, after finishing training mode, the front-end will transition to batch query mode instead of query mode. In batch query mode, all queries are submitted at once, and the front-end will indicate when the queries have finished before any results are returned.

## Commands

### Configuration mode

#### `frontend batch-queries V` (three tokens)

If `V` is `1`, then request that the front-end transition into batch query mode, and not query mode, after training mode has finished. If `V` is anything else, then request that it transition into query mode as usual.

Responses:

* `epbprtv0 ok`

  The front-end will transition into the requested query mode after the training mode has finished.

* `epbprtv0 fail`

  This command has had no effect on the query mode transition.

### Training mode

This extension changes the behaviour of one command in training mode:

#### *empty line* (zero tokens)

Finish training mode and enter batch query mode.

Responses:

* `epbprtv0 ok COUNT1 [fail COUNT2]`

  `COUNT1` (potentially zero) entries were successfully interpreted and added to the data structure. (`COUNT2` entries couldn't be interpreted or couldn't be added for other reasons.):

### Batch query mode

In batch query mode, front-ends should respond to three different kinds of command:

#### `ENTRY0 [..., ENTRYk] N` (two or more tokens)

Prepare to run a query to find at most `N` (greater than or equal to 1) close matches for each of the `k` query points from `ENTRY0` to `ENTRYk`.

Responses:

* `epbprtv0 ok`

  Preparation is complete, and the `query` command can now be used.

* `epbprtv0 fail`

  Preparation has failed, and the `query` command should not be used. This may occur if one of the `k` query points could not be parsed.

#### `query` (one token)

Run the last prepared query.

Responses:

* `epbprtv0 ok`

  The query was executed successfully. `k` sets of results will appear after this line, each of them of the same form as in the normal query mode.

* `epbprtv0 fail`

  No query has been prepared.

#### *empty line* (zero tokens)

Finish prepared query mode and terminate the front-end.

Responses:

* `epbprtv0 ok`

  The front-end has terminated.
