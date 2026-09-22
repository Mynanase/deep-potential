#import "@preview/touying:0.7.4": *
#import themes.metropolis: *

#import "@preview/numbly:0.1.0": numbly

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [MW Halo Model with Normalizing Flow],
    subtitle: [],
    author: [Qiu Tao],
    date: datetime.today(),
    institution: [SHAO],
    contact: [qiutao\@shao.ac.cn],
    // logo: emoji.city,
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

#outline-slide()

= Overview




= Auriga Data

== Removal unstable structure

