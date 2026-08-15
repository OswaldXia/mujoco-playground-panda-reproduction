# 运行产物与保存策略

## 三层保存

1. **Git：** 源码、脚本、测试、说明、小型 machine-readable summary；
2. **服务器 artifact：** checkpoint、events、完整 JSON/CSV、console、视频；
3. **外部归档：** 完整 artifact tar.gz 与 SHA-256，防服务器清理。

普通 Git 不适合频繁变化的大型二进制 checkpoint/video。若未来公开 checkpoint，
使用 GitHub Release 或对象存储，并在 Git 文档中记录 URL、大小、commit、训练
协议与 SHA-256；不要解除 `artifacts/.gitignore` 后直接提交。

## 训练后最低清单

- [ ] `manifest.json`：版本、设备、commit、dirty；
- [ ] `console.log`：完整命令输出和错误；
- [ ] `evaluation-summary.json`：每个 evaluation boundary；
- [ ] checkpoint 目录及能恢复的最后/最佳步；
- [ ] TensorBoard event；
- [ ] 成功与失败 replay；
- [ ] fine-tune source/checkpoint selection 记录；
- [ ] 非覆盖归档与 checksum。

## 正式评估后最低清单

- [ ] 完整 schema-4 evaluation JSON；
- [ ] effective guide probability 0.0；
- [ ] 每回合 record、episodes.csv；
- [ ] aggregate、per-seed、Wilson interval、worst seed；
- [ ] position bins/plot/failure cases；
- [ ] trajectory failure classification；
- [ ] acceptance 原始门槛与 PASS/FAIL；
- [ ] compact summary 提交 Git，原始大文件保留 artifact/归档。

## 典型目录

```text
reproduction/artifacts/
├── panda-vision-smoke/
├── panda-vision-full/
├── panda-vision-finetune/
├── panda-independent-eval/
├── panda-vision-robustness/
├── panda-robustness-eval/
└── panda-failure-modes/
```

教程中的 `docs/assets/` 只放经过选择、体积受控、具有解释价值的静态素材。
