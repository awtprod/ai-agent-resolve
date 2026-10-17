# DRP test fixtures

These four project files are copied unchanged from samuelgursky/davinci-resolve-mcp, directory
`resolve-advanced/vendor/drp-format/templates/`, at commit
`90eba8cdf4df333beb4ce41fe0ab198b8fa7d125`. They are released under the MIT License; the
license text is in [`LICENSE`](LICENSE), copied from the root of that repository at the same
commit.

| File | Git blob SHA (matches upstream) | Resolve stamp |
| ---- | ------------------------------- | ------------- |
| `empty-project.drp` | `8575ee9faa1699289b689e1c0646e8702862b725` | 21.0.0.0048, DbPrjVer 17 |
| `empty-project-r19.drp` | `5e2aa4ee8c07ce26cafb4f21fd1b3effc70e1125` | 19.1.3.0007, DbPrjVer 14 |
| `media-clip-h264.drp` | `3a06501b6deb2f957fc823ceb412eaa37a8b0d18` | 21.0.0.0048, DbPrjVer 17 |
| `media-clip-r19.drp` | `4e0a34380e42aaa269985ebb95731a6b0c3f00f6` | 19.1.3.0007, DbPrjVer 14 |

Real project files must not be added here. They go in the git-ignored `corpus/` directory or
in `$AAR_DRP_CORPUS`, which `tests/drp/test_corpus.py` reads when it is set.
