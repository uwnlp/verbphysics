# verbphysics

## about

This repository contains the data and reference implementation for the paper
_Verb Physics: Relative Physical Knowledge of Actions and Objects_ by Maxwell
Forbes and Yejin Choi, published at ACL2017.

See the [project page](https://uwnlp.github.io/verbphysics/) for more details.

## installation / running

This section is under construction as we setup Travis-CI for a reproducible
build.

## data

The verbphysics data is found under `data/`.

**Task setup as in the ACL 2017 paper:**

When predicting action frames, only 5% action frame data should be used. Either
5% (our model A) or 20% object pair data (our model B) may be used to assist in
action frame prediction.

When predicting object pairs, only 5% object pair data should be used. Either 5%
(our model A) or 20% action frame data (our model B) may be used to assist in
object pair prediction.