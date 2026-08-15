module hann_bridge

go 1.23.0

require github.com/habedi/hann v0.0.0-00010101000000-000000000000

// The Dockerfile clones the hann repository into hann-src next to this file.
// For a local build, point the replace directive at a hann working copy, or
// place one at hann-src.
replace github.com/habedi/hann => ./hann-src
