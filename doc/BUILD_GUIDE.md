# 跨平台打包指南

由于 macOS 和 Windows 的底层架构不同（Mach-O vs PE 格式），**无法直接在 macOS 上生成 Windows 的 .exe 可执行文件**。

为了解决这个问题，我们为您配置了 **GitHub Actions 自动化构建流程**。这是目前最标准、最简便的解决方案。

## 方案一：使用 GitHub Actions 自动构建（推荐）

您只需要将代码推送到 GitHub，云端服务器会自动为您编译 Windows 版本。

### 步骤：
1. **上传代码**: 将当前项目上传到您的 GitHub 仓库。
2. **触发构建**:
   - 每次 `push` 代码时会自动触发。
   - 或者在 GitHub 仓库页面的 "Actions" 标签页手动触发 "Build Windows Exe"。
3. **下载软件**:
   - 构建完成后（约 3-5 分钟），进入该次 Action 的详情页。
   - 在页面底部的 **Artifacts** 区域，点击 `ImagePerfusion-Windows` 即可下载 `.exe` 文件。

### 配置文件位置：
- `.github/workflows/build_windows.yml`: 定义了 Windows 构建流程。

---

## 方案二：在 Windows 虚拟机中构建

如果您有 Windows 电脑或虚拟机（如 Parallels Desktop），可以手动构建：

1. **安装 Python**: 确保安装 Python 3.9+。
2. **安装依赖**:
   ```cmd
   pip install -r src/requirements.txt
   pip install pyinstaller
   ```
3. **执行打包命令**:
   ```cmd
   pyinstaller --noconfirm --onefile --windowed --name "ImagePerfusion" --collect-all tkinterdnd2 --add-data "src;src" run_gui.py
   ```
4. **获取文件**: 在 `dist/` 文件夹中找到 `ImagePerfusion.exe`。

## 方案三：构建 macOS 本地版本

如果您想在当前 Mac 上打包为 App：

```bash
# 安装 pyinstaller
pip install pyinstaller

# 执行打包
pyinstaller --noconfirm --onefile --windowed --name "ImagePerfusion" --collect-all tkinterdnd2 --add-data "src:src" run_gui.py
```
*注意：macOS 上 `--add-data` 使用冒号 `:` 分隔，Windows 使用分号 `;` 分隔。*
