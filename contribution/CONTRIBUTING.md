# __Contribution Guidelines__
Welcome to _Evalverse_! We warmly welcome any kind of contribution 😊✨. </br>
This page provides an outline on how to contribute to _Evalverse_ and suggestions for nice conventions to follow. 
> __These are guidelines, NOT rules 💡__ <p>
This page is not the Constituion of the _Evalverse_. We are providing guidelines to help you make a useful and efficient contribution to _Evalverse_. While we think these guidelines are sensible and we appreciate when they are observed, following them isn't strictly required. We hope you won't be tired by these guidelines. Also, we'd love to hear your ideas on how to improve our guidelines! 

</br>

# Table of Contents
- [Questions or Feedback](#questions-or-feedback)
- [🤝 How to Contribute?](#how-to-contribute)
- [Commit Guidelines](#commit-guidelines)
- [Style Guides](#style-guides)

</br>

# Questions or Feedback
Join the conversation on our GitHub discussion board! It's the go-to spot for questions, chats, and a helping hand from the _Evalverse_ community. Drop by and say hello here: [link](https://github.com/UpstageAI/evalverse/discussions)

And if there's a shiny new feature you're dreaming of, don't be shy—head over to our [issue page](https://github.com/UpstageAI/evalverse/issues) to let us know! Your input could help shape the future. ✨

</br>

# How to Contribute?
- Any kind of improvement of document: fixing typo, enhancing grammar or semantic structuring or adding new examples.
- Submit issues related to bugs, new desired features, or enhancement of existing features.
- Fix a bug, implement new feature or improving existing feature.
- Answer other users' question or help.


## __Report a Bug / Request New Feature / Suggest Enhancements__
Please open an issue whenever you find a bug or have an idea to enhance _Evalverse_. Maintainers will label it or leave comment on it as soon as they check the issue. Issues labeled as `Open for contribution` mean they are open for contribution.

## __Fix a Bug / Add New Feature / Improve Existing Feature__
If you have a particular roadmap, goals, or new feature, share it via issue. already fixed a bug or have new feature that enhances _Evalverse_, you can jump on to fourth step which is opening pull requests. Please note that when you open pull requests without opening an issue or maintainers' check, it can be declined if it does not aligh with philosophy of _Evalverse_.

### __1️⃣ Check issues labeled as__ `Open for contribution`
You can find issues waiting for your contribution by filtering label with `Open for contribution`. This label does not stand alone. It is always with `Bug`, `Docs` or `Enhancement`. Issues with `Critical` or `ASAP` label are more urgent. 


### __2️⃣ Leave a comment on the issue you want to contribute__
Once we review your comment, we'll entrust the issue to you by swapping out the `Open for contribution` label for a `WIP` (Work in Progress) label.

### __3️⃣ Work on it__
Before diving into coding, do take a moment to familiarize yourself with our coding style by visiting this [style guides](#style-guides). And hey, if you hit a snag while tackling the issue, don't hesitate to drop a comment right there. Our community is a supportive bunch and will jump in to assist or brainstorm with you.

1. Fork the repository of _Evalverse_.
2. Clone your fork to your local disk.
3. Create a new branch to hold your develompment changes. </br>
It's not required to adhere strictly to the branch naming example provided; consider it a mild suggestion.
```bash
git checkout -b {prefix}/{issue-number}-{description}
```
4. Set up a development environment
5. Develop the features in your branch


### __4️⃣ Create a Pull Request__
Go ahead and visit your GitHub fork, then initiate a pull request — it's time to share your awesome work! Before you do, double-check that you've completed everything on the checklist we provided. Once you're all set, submit your contributions for the project maintainers to review.

Don't worry if the maintainers have some feedback or suggest changes—it's all part of the process and happens to even our most experienced contributors. Keep your updates flowing by working in your local branch and pushing any new changes to your fork. Your pull request will update automatically for everyone to see the progress.

</br>

# Commit Guidelines
### Commit strategy
- Avoid mixing multiple, unrelated modifications in a single commit. One commit is related with one issue.
- Each commit should encapsulate a complete, autonomous upgrade to the code.

### Commit messages
Please make sure your commit messages follow `type`: `title (#<related issue number>)` format. <br/>
For example:
```plain text
<TYPE>: Short summary with 72 characters or less (#<Issue number>)

If you have more detalied explanatory text, put it as body.
But the body is optional.
```
- Find adequate type in the below list:
    - `NEW`: introducing a new feature
    - `ENHANCE`: improve an existing code/feature.
    - `FIX`: fix a code bug
    - `DOCS`: write/update/add any kind of documents including docstring
    - `REFACTOR`: refactor existing code without any specific improvements
    - `STYLE`: changes that do not affect the meaning of the code (ex. white-space, line length)
    - `TEST`: add additional testing
    - `DEL`: remove code or files
    - `RELEASE`: release new version of evalverse
    - `OTHER`: anything not covered above (not recommended)
- Use the present tense ("Add feature" not "Added feature")
- Do not end the subject line with a punctuation

</br>

# Style Guides
### Pre-commit hook
We provide a pre-commit git hook for style check. You can find exact check list in this [file](https://github.com/UpstageAI/evalverse/blob/main/.pre-commit-config.yaml). <br/> Please run the code below before a commit is created:
```bash
pre-commit run
```

