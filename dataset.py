import random
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder


class TripletFolder(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super(TripletFolder, self).__init__(root=root, transform=transform)

        # Create a dictionary of lists for each class for reverse lookup
        # to generate triplets
        self.classdict = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdict[ci] = []

        # append each file in the approach dictionary element list
        for s in self.samples:
            self.classdict[s[1]].append(s[0])

        # keep track of the sizes for random sampling
        self.classdictsize = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdictsize[ci] = len(self.classdict[ci])

    # Return a triplet, with positive and negative selected at random.
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, sample, sample) where the samples
                    are anchor, positive, and negative.
            The positive and negative instances are sampled randomly.
        """

        # The anchor is the image at this index.
        a_path, a_target = self.samples[index]

        prs = random.random()  # positive random sample
        nrc = random.random()  # negative random class
        nrs = random.random()  # negative random sample

        # random negative class cannot be the same class as anchor. We add
        # a random offset that is 1 less than the number required to wrap
        # back around to a_target after modulus.
        nrc = (a_target + int(nrc * (len(self.classes) - 1))) % len(self.classes)

        # Positive Instance: select a random instance from the same class as anchor.
        p_path = self.classdict[a_target][int(self.classdictsize[a_target] * prs)]

        # Negative Instance: select a random instance from the random negative class.
        n_path = self.classdict[nrc][int(self.classdictsize[nrc] * nrs)]

        # Load the data for these samples.
        a_sample = self.loader(a_path)
        p_sample = self.loader(p_path)
        n_sample = self.loader(n_path)

        # note that we do not return the label!
        return a_sample, p_sample, n_sample
