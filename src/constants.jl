
#   This file is part of TensorFields.jl. It is licensed under the GPL license
#   TensorFields Copyright (C) 2019 Michael Reed

const Gauss = Values(
    Values(
        Values(1),
        Values(
            Values(1/3,1/3))),
    Values(
        Values(1,1,1)/3,
        Values(
            Values(1/6,1/6),
            Values(2/3,1/6),
            Values(1/6,2/3))),
    Values(
        Values(-27/48,25/48,25/48),
        Values(
            Values(1/3,1/3),
            Values(1/5,1/5),
            Values(3/5,1/5),
            Values(1/5,3/5))),
    Values(
        Values(35494641/158896895,35494641/158896895,40960013/372527180,40960013/372527180,40960013/372527180),
        Values(
            Values(100320057/224958844,100320057/224958844),
            Values(100320057/224958844,16300311/150784976),
            Values(16300311/150784976,100320057/224958844),
            Values(13196394/144102857,13196394/144102857),
            Values(13196394/144102857,85438943/104595944),
            Values(85438943/104595944,13196394/144102857))))
