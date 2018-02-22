(This document describes an extension that front-ends aren't required to implement. Front-ends that don't implement this extension should reject attempts to set the `prepared-queries` front-end configuration option.)

When the front-end configuration option `prepared-queries` is set to `1`, after finishing training mode, the front-end will transition to prepared query mode instead of query mode. In prepared query mode, parsing a query point -- a potentially expensive operation -- and actually running a query are two different commands; this makes the query timings more representative of the underlying algorithm's behaviour without the overhead of this protocol.

## Commands

### Prepared query mode

In prepared query mode, front-ends should respond to three different kinds of command:

#### `ENTRY N` (two tokens)

Prepare to run a query to find at most `N` (greater than or equal to 1) close matches for `ENTRY`.

Responses:

* `epbprtv0 ok prepared true`

  Preparation is complete, the `query` command can now be used, and the underlying library wrapper has special support for prepared queries.

* `epbprtv0 ok prepared false`

  The `query` command can now be used, but the underlying library wrapper doesn't have support for prepared queries, so the `query` command will perform the parsing of `ENTRY` as it would in normal query mode.

#### `query` (one token)

Run the last prepared query.

Responses:

* `epbprtv0 ok R`

  `R` (greater than zero and less than or equal to the value of `N` that was specified when the query was prepared) close matches were found. The next `R` lines, when tokenised, will consist of the token `epbprtv0` followed by a token specifying the index of a close match. (The first line should identify the *closest* close match, and the `R`-th should identify the furthest away.)

* `epbprtv0 fail`

  Either no close matches were found, or no query has been prepared.

#### *empty line* (zero tokens)

Finish prepared query mode and terminate the front-end.

Responses:

* `epbprtv0 ok`

  The front-end has terminated.
