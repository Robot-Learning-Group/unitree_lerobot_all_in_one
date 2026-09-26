#!/usr/bin/env bash
set -euo pipefail

# 日本語・英語の案内を表示する。端末以外とNO_COLOR指定時は無色。
# Show bilingual messages; omit ANSI colors for redirected output or NO_COLOR.
message() {
  local level="$1" ja="$2" en="$3"
  local icon color='' reset='' code
  case "$level" in
    success) icon='✅'; code=32 ;;
    warning) icon='⚠️'; code=33 ;;
    error)   icon='❌'; code=31 ;;
    *)       icon='ℹ️'; code=36 ;;
  esac
  if [[ -t 1 && "${TERM:-dumb}" != dumb && -z "${NO_COLOR+x}" ]]; then
    printf -v color '\033[%sm' "$code"
    printf -v reset '\033[0m'
  fi
  printf '%s%s %s\n   %s%s\n' "$color" "$icon" "$ja" "$en" "$reset"
}

if [[ "${EUID}" -ne 0 ]]; then
  message error "管理者権限が必要です。sudo bash \"$0\" で実行してください。" "Run this script with: sudo bash \"$0\"" >&2
  exit 1
fi

. /etc/os-release
if [[ "${ID}" != "ubuntu" ]]; then
  message error "Ubuntu用のスクリプトです。検出したOS：${PRETTY_NAME}" "This script expects Ubuntu. Detected: ${PRETTY_NAME}" >&2
  exit 1
fi

# CUDA 12.8 repository selection for Ubuntu x86_64 workstations.
case "${VERSION_ID}:$(dpkg --print-architecture)" in
  20.04:amd64|22.04:amd64|24.04:amd64)
    cuda_repo="ubuntu${VERSION_ID//./}/x86_64"
    ;;
  *)
    message error "対応OSはUbuntu 20.04／22.04／24.04のx86_64版です。" "CUDA 12.8 setup supports Ubuntu 20.04/22.04/24.04 x86_64 in this script." >&2
    exit 1
    ;;
esac

# Check the loaded module, not just an installed package or nvidia-smi's version.
# Installing a driver does not replace the module already running in the kernel.
if [[ -r /proc/driver/nvidia/version ]] &&
   grep -Eq '^NVRM version: NVIDIA UNIX Open Kernel Module for x86_64[[:space:]]+580\.' /proc/driver/nvidia/version; then
  message success "NVIDIA 580系のopenカーネルモジュールが稼働しています。" "NVIDIA 580 open kernel module is running."
else
  message warning "NVIDIA 580系のopen版が稼働していません。インストール条件を確認します。" "NVIDIA 580 open kernel module is not running. Checking installation requirements."
  if [[ -r /proc/driver/nvidia/version ]]; then
    cat /proc/driver/nvidia/version
  fi

  # A runfile installation must be removed with its own uninstaller first.
  if command -v nvidia-uninstall >/dev/null 2>&1; then
    message error ".run版のアンインストーラーが見つかりました。APTへ切り替える前に、そのドライバを削除してください。" "An NVIDIA runfile uninstaller was found. Remove that driver installation before switching to APT." >&2
    exit 1
  fi

  apt-get update
  driver_candidate="$(LC_ALL=C apt-cache policy nvidia-driver-580-open | awk '/^[[:space:]]*Candidate:/ {print $2; exit}')"
  case "${driver_candidate#*:}" in
    580.*) ;;
    *)
      message error "APT配布元に580系の候補がありません。検出した候補：${driver_candidate:-none}" "No 580-series candidate for nvidia-driver-580-open in the configured APT repositories (candidate: ${driver_candidate:-none})." >&2
      message warning "Ubuntuのバージョン、配布元、GPUの対応状況を確認してください。" "Check this Ubuntu release, repositories, and GPU support before continuing." >&2
      exit 1
      ;;
  esac

  # Let APT replace conflicting driver packages; do not purge all NVIDIA tools.
  message info "nvidia-driver-580-openをインストールします。" "Installing nvidia-driver-580-open."
  apt-get install -y nvidia-driver-580-open
  message success "NVIDIA 580 openを導入しました。CUDA・Dockerの設定は再起動後に行います。" "NVIDIA 580 open driver package installed. CUDA/Docker setup will run after reboot."
  message warning "Secure Bootの鍵登録を求められた場合は、再起動時に登録してください。" "If Secure Boot key enrollment was requested, complete it during reboot."
  message warning "再起動後、同じスクリプトをsudo bashで再実行してください。" "Reboot this machine, then run this same script again with sudo bash."
  exit 0
fi

# A matching module must also communicate with the installed userspace driver.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  message error "580 openは読み込み済みですが、nvidia-smiがありません。ドライバを修復してください。" "NVIDIA 580 open module is loaded, but nvidia-smi is missing. Repair the driver installation first." >&2
  exit 1
fi
if ! nvidia-smi; then
  message error "nvidia-smiが失敗しました。ドライバ変更直後なら再起動し、それ以外ならドライバを修復してください。" "nvidia-smi failed. Reboot if the driver was just changed; otherwise repair the driver installation." >&2
  exit 1
fi

target_user="${SUDO_USER:-}"
if [[ -z "${target_user}" || "${target_user}" == "root" ]]; then
  target_user="$(logname 2>/dev/null || true)"
fi

message info "CUDA・Dockerのセットアップに必要な基本パッケージとGit・Git LFSをインストールします。" "Installing prerequisite packages for CUDA and Docker, plus Git and Git LFS."
apt-get update
apt-get install -y ca-certificates curl gnupg build-essential git git-lfs

# Register the LFS filters system-wide so every user's clone fetches LFS files.
git lfs install --system
message success "Git LFSを有効化しました。" "Git LFS is enabled."

# Install only the CUDA 12.8 Toolkit; do not request driver packages.
message info "CUDA Toolkit 12.8をインストールします。" "Installing CUDA Toolkit 12.8."
cuda_tmpdir="$(mktemp -d)"
trap 'rm -rf -- "${cuda_tmpdir}"' EXIT
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/cuda-keyring_1.1-1_all.deb" \
  -o "${cuda_tmpdir}/cuda-keyring.deb"
dpkg -i "${cuda_tmpdir}/cuda-keyring.deb"
apt-get update
apt-get install -y cuda-toolkit-12-8

# Make the selected toolkit available in subsequent login shells.
cat > /etc/profile.d/cuda-12-8.sh <<'CUDA_ENV'
export CUDA_HOME=/usr/local/cuda-12.8
case ":${PATH}:" in
  *":${CUDA_HOME}/bin:"*) ;;
  *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
esac
CUDA_ENV
chmod 0644 /etc/profile.d/cuda-12-8.sh
# Also load CUDA for interactive non-login Bash shells (new terminals).
if [[ -n "${target_user}" ]]; then
  target_home="$(getent passwd "${target_user}" | cut -d: -f6)"
  if [[ -z "${target_home}" || ! -d "${target_home}" ]]; then
    message error "${target_user}のホームディレクトリが見つかりません。" "Could not find the home directory for ${target_user}." >&2
    exit 1
  fi
  target_bashrc="${target_home}/.bashrc"
  if [[ ! -e "${target_bashrc}" ]]; then
    install -m 0644 -o "${target_user}" -g "$(id -gn "${target_user}")" /dev/null "${target_bashrc}"
  fi
  cuda_bashrc_line='[ ! -r /etc/profile.d/cuda-12-8.sh ] || . /etc/profile.d/cuda-12-8.sh'
  if ! grep -qxF -- "${cuda_bashrc_line}" "${target_bashrc}"; then
    printf '\n# CUDA Toolkit 12.8\n%s\n' "${cuda_bashrc_line}" >> "${target_bashrc}"
  fi
  message success "${target_bashrc}にCUDA 12.8の設定を追加しました。" "Configured CUDA 12.8 in ${target_bashrc}."
  message info "現在開いているユーザー端末で反映するには、source ~/.bashrcを実行してください。" "In your existing user terminal, run: source ~/.bashrc"
else
  message warning "ログインユーザーを特定できず、.bashrcを更新できませんでした。" "Could not detect login user; .bashrc was not updated." >&2
  message info "普段使うユーザーからsudo bashでこのスクリプトを実行してください。" "Run this script via sudo bash from your regular user account." >&2
fi

. /etc/profile.d/cuda-12-8.sh
nvcc --version

message info "Docker Engine・Buildx・Composeをインストールします。" "Installing Docker Engine, Buildx, and Compose."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

message info "NVIDIA Container Toolkitをインストールします。" "Installing NVIDIA Container Toolkit."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
  gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
message warning "GPU設定を反映するため、Dockerサービスを再起動します。" "Restarting the Docker service to apply the GPU configuration."
systemctl restart docker

if [[ -n "${target_user}" ]]; then
  usermod -aG docker "${target_user}"
  message success "${target_user}をdockerグループに追加しました。" "Added ${target_user} to docker group."
else
  message warning "ログインユーザーを特定できません。管理者権限でusermod -aG docker USERを実行してください（USERは対象ユーザー名）。" "Could not detect login user. Add your user to docker group manually: usermod -aG docker USER"
fi

message info "DockerとGPUの動作を確認します。コンテナ内でもnvidia-smiを実行します。" "Checking Docker and GPU access, including nvidia-smi inside a container."
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi

message success "セットアップが完了しました。CUDA Toolkit 12.8とDockerからのGPU認識を確認しました。" "Setup complete. CUDA Toolkit 12.8 and GPU visibility from Docker have been verified."
message info "CUDAのPATHとsudoなしのDocker利用を反映するため、ログアウトして再ログインしてください。" "Log out and back in for CUDA PATH and non-root docker access."
