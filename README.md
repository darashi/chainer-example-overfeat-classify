# chainer-example-overfeat-classify

Image classifier built with [Chainer][], an implementation of [OverFeat][].

## Requirements

* [Chainer][]

## Install

Download pre-trained models:
```sh
sh download_model.sh
```

## Run


Classify `example.jpg` with `fast` model on CPU:

```sh
python classify.py example.jpg
```

Classify `example.jpg` with `accurate` model on CPU:

```sh
python classify.py --model accurate example.jpg
```

Classify `example.jpg` with `fast` model on GPU (ID=0):

```sh
python classify.py --gpu 0 example.jpg
```

## References

* [Chainer][]
* [OverFeat][]
* [overfeat-torch][]

  [Chainer]: https://github.com/pfnet/chainer
  [OverFeat]: https://github.com/sermanet/OverFeat
  [overfeat-torch]: https://github.com/jhjin/overfeat-torch
