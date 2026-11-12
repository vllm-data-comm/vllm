# LWD Frontend Attachment

`vllm/entrypoints/openai/lwd/frontend_attach.py` gives every edge in an
`edge_share` topology its own API frontend while sharing the primary edge's
native engine group. The primary edge is client index 0 and is the only command
that launches engines. Each other edge runs one API-server child and attaches
to the primary with `--api-server-attach`; the primary listens with
`--api-server-rpc-port`.

## Handshake

The attachment control channel is separate from the LWD engine control plane.
It carries msgpack dictionaries over a ZMQ ROUTER/DEALER connection:

1. The non-primary frontend sends `LWD_FRONTEND_ATTACH` with its edge-derived
   `client_index`, explicit engine-affecting configuration, API listen address,
   and pid. The primary rejects an invalid or duplicate index, incompatible
   configuration, or an API listen-address conflict. Local listen, HTTP
   transport, and attachment plumbing options are excluded from comparison;
   serving-only options are inherited when not explicitly set by the child.
2. `LWD_FRONTEND_ACCEPT` returns the topology-derived client count, the primary
   stats-update address, and inheritable frontend arguments.
3. The child starts its ordinary API server with ephemeral native input/output
   endpoints, then sends their actual bound addresses in
   `LWD_FRONTEND_ENDPOINTS`.
4. The primary validates and records those addresses and answers
   `LWD_FRONTEND_BARRIER`. Errors at any stage use `LWD_FRONTEND_ERROR`.

The client index and count come from the resolved LWD topology, not process
start order. `--api-server-count` must equal the number of topology edges.

## Startup barrier

The primary starts only local frontend index 0. While the
`launch_core_engines` context is still open, it collects every other index's
native ROUTER/PULL endpoints and writes them into the shared engine address
arrays. It also fails registration if an engine process exits, and times out
with the missing client indices. Leaving the context then performs the native
engine handshake with the complete frontend set. Thus no engine handshake can
pass before all required frontends are registered.

## `client_index` output routing

The native request path already stamps each request with its originating
`request.client_index`. LWD does not add a second frontend-routing protocol:
engine output is sent through the output socket at that same index, returning
it to the frontend that accepted the request. The primary always occupies slot
0; attached frontends occupy their topology edge IDs in slots 1 through N-1.
