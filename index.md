# Daniel Melo
Welcome to my personal portfolio. Here you could find some of my repos, posts and investigation around my CS degree 

## Investigation
- [The zero behind math](https://dmeloca.github.io/investigation/zero-behind-math.html)

## Blog
- [Cs degree vision](https://dmeloca.github.io/posts/cs-degree-vision.html)
- [PyTorch Notes](https://dmeloca.github.io/posts/pytorch.html)
- [Gamification](https://dmeloca.github.io/posts/gamification.html)
## Projects
{% for repository in site.github.public_repositories %}
  * [{{ repository.name }}]({{ repository.html_url }})
{% endfor %}


