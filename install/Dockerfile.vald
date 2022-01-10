ARG VALD_VERSION=v1.3.1
FROM vdaas/vald-agent-ngt:${VALD_VERSION} as vald

FROM ann-benchmarks
ARG VALD_CLIENT_VERSION=1.3.1
COPY --from=vald /go/bin/ngt /go/bin/ngt

RUN pip3 install vald-client-python==${VALD_CLIENT_VERSION}
