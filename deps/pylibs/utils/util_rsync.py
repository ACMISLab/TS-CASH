import os.path

from pylibs.utils.util_bash import BashUtil
from pylibs.utils.util_servers import Server


class RSYNCOption:
    # 只同步 .py 和 .csv
    INCLUDE_ONLY_PY_AND_CSV = "--include='*.csv' --include='*.py' --exclude='*'"


class UtilRsync:
    """
    选项:
    --progress 显示进度条
    """

    def __init__(self, local_home, remote_home, server_conf: Server):
        self.conf = server_conf
        self.local_home = local_home
        self.remote_home = remote_home

    def upload(self, dir_name):
        """
        Upload a directory

        Parameters
        ----------

        Returns
        -------
        """
        # Fdd19950518*

        # sshpass -p Fdd19950518* rsync -avi  -e "ssh -p 20045" root@your_server_ip:/remote-home/acmis_fdd/sw_research_code/runtime/ ./
        _r = os.path.join(self.remote_home, dir_name)
        _l = os.path.join(self.local_home, dir_name)
        cmd = f'sshpass -p "{self.conf.password}" rsync  -e "ssh -p {self.conf.port}" -avi {_l}/  root@{self.conf.ip}:{_r}/'
        return BashUtil.run_command_print_progress(cmd)

    def upload_file(self, file_name):
        """
        Upload a directory

        Parameters
        ----------

        Returns
        -------

        """
        remote_file = os.path.join(self.remote_home, file_name)
        cmd = f'sshpass -p "{self.conf.password}" rsync  -e "ssh -p {self.conf.port}" -avi {file_name} root@{self.conf.ip}:{remote_file}'
        BashUtil.run_command_print_progress(cmd)
        return remote_file

    def download_file(self, remote_path, local_paht):
        """
        Upload a directory

        Parameters
        ----------

        Returns
        -------

        """
        cmd = f'sshpass -p "{self.conf.password}" rsync  -e "ssh -p {self.conf.port}" -avi root@{self.conf.ip}:{remote_path} {local_paht} '
        BashUtil.run_command_print_progress(cmd)

    def download(self, dir_name=""):
        """
        Download a directory
        Parameters
        ----------
        dir_name : the dir to download

        Returns
        -------

        """
        _r = os.path.join(self.remote_home, dir_name)
        _l = os.path.join(self.local_home, dir_name)
        cmd = f'sshpass -p "{self.conf.password}" rsync  -e "ssh -p {self.conf.port}" -avi  root@{self.conf.ip}:{_r}/ {_l}/'
        return BashUtil.run_command_print_progress(cmd)


class Rsync:
    @staticmethod
    def _pre_process_dir(local):
        if not local.endswith("/"):
            return local + "/"
        return local

    SYNC_OPTIONS = ' --debug=FILTER --max-size=10M  -f"+ p02/***" -f"- *.log" -f"+ Makefile" -f"+ *.sh" -f"+ A01_paper_exp/cache.csv" -f"- __pycache__/" -f"- .git/" -f"- *.csv" -f"- *.zip" -f"- lightning_logs/" -f"- aiops2018/" -f"- *.png" -f"- runtime/" -f"+ *.py" -f"+ A01_paper_exp/"  -f"+ libs/" -f"+ libs/datasets/***"  -f"+ A01_paper_exp/tests/***" -f"+ libs/py-search-lib/***" -f"+ libs/timeseries-models/***"  -f"- *" '

    SSH_OPTIONS = " -o PubkeyAuthentication=yes   -o stricthostkeychecking=no "

    @classmethod
    def generate_upload_cmd(cls, conf: Server, local, remote, sync_option=None):

        if sync_option is None:
            sync_option = Rsync.SYNC_OPTIONS

        return f'sshpass -p {conf.password} rsync  -e "ssh -p {conf.port} {Rsync.SSH_OPTIONS}" -avi  {sync_option}  --exclude=".git/" {local} {conf.username}@{conf.ip}:{remote}'

    @classmethod
    def _generate_download_cmd(cls, conf: Server, local, remote, options=""):
        return f'sshpass -p "{conf.password}" rsync  -e "ssh -p {conf.port} {Rsync.SSH_OPTIONS}" -avi {options}  {conf.username}@{conf.ip}:{remote} {local}'

    @staticmethod
    def upload(conf: Server, local):
        local = Rsync._pre_process_dir(local)
        remote = Rsync._pre_process_dir(conf.work_home)
        cmd = Rsync.generate_upload_cmd(conf, local, remote)
        BashUtil.run_command_print_progress(cmd)

    @staticmethod
    def upload_dir(conf: Server, local, remete, sync_options=None):
        local = Rsync._pre_process_dir(local)
        remote = Rsync._pre_process_dir(remete)
        cmd = Rsync.generate_upload_cmd(conf, local, remote, sync_options)
        BashUtil.run_command_print_progress(cmd)

    @staticmethod
    def upload_file(server: Server, local_file, remote_file=None):
        """
        使用rsync上传单个文件到server
        Parameters
        ----------
        server :
        local_file :
        remete :
        sync_options :

        Returns
        -------

        """
        assert server.data_home is not None, "Server.data_home can't be None"
        if remote_file is None:
            remote_file = os.path.join(server.data_home, os.path.basename(local_file))

        local = Rsync._pre_process_file(local_file)
        remote = Rsync._pre_process_file(remote_file)
        print(f"🚗 Upload {local} to {server.ip}{remote}")
        # cmd = Rsync.generate_upload_cmd(conf, local, remote, sync_options)
        cmd = Rsync._generate_upload_cmd_from_server(server, local, remote, sync_option="")
        BashUtil.run_command_print_progress(cmd)

    @staticmethod
    def download_file(server: Server, remote_file, local_file):
        """
        下载文件
        Parameters
        ----------
        server :
        remete :
        local_file :
        sync_options :

        Returns
        -------

        """
        local = Rsync._pre_process_file(local_file)
        remote = Rsync._pre_process_file(remote_file)
        # print(f"Download  {server.ip}:{remote} to {local}")
        cmd = Rsync._generate_download_cmd(server, local, remote, options="")
        print(cmd)
        BashUtil.run_command_print_progress(cmd)

    @staticmethod
    def download_dir(conf: Server, local, remote, options=""):
        local = Rsync._pre_process_dir(os.path.join(local, os.path.basename(remote)))
        remote = Rsync._pre_process_dir(remote)
        cmd = Rsync._generate_download_cmd(conf, local, remote, options=options)
        BashUtil.run_command_print_progress(cmd)

    @classmethod
    def check_is_success(cls, param):
        pass

    @classmethod
    def _generate_upload_cmd_from_server(cls, server, local, remote, sync_option):
        """
        从server生产上传命令，需要提前安装rsync

        Parameters
        ----------
        server :
        local :
        remote :
        sync_option :

        Returns
        -------

        """
        if sync_option is None:
            sync_option = Rsync.SYNC_OPTIONS

        return f'sshpass -p {server.password} rsync  -e "ssh -p {server.port} {Rsync.SSH_OPTIONS}" -avi  {sync_option}  --exclude=".git/" {local} {server.username}@{server.ip}:{remote}'

    @classmethod
    def _pre_process_file(cls, file_name):
        return os.path.abspath(file_name)


class UtilRsyncV2:
    def __init__(self, source: Server, target: Server):
        self.source = source
        self.target = target

    def get_remote_home(self):
        return self.target.home

    def upload_dir(self, dir_name):
        """
        Upload a directory

        Parameters
        ----------

        Returns
        -------
        """
        # sshpass -p Fdd19950518* rsync -avi  -e "ssh -p 20045" root@your_server_ip:/remote-home/acmis_fdd/sw_research_code/runtime/ ./
        _l = os.path.join(self.source.home, dir_name)
        _r = os.path.join(self.target.home, dir_name)
        cmd = f'sshpass -p "{self.target.password}" rsync  -e "ssh -p {self.target.port}" -avi {_l}/  root@{self.target.ip}:{_r}/'
        return BashUtil.run_command_print_progress(cmd)

    def upload_file(self, file_name):
        """
        Upload a directory

        Parameters
        ----------

        Returns
        -------

        """
        remote_file = os.path.join(self.target.home, file_name)
        # local_file = os.path.abspath(os.path.join(self.source.home, file_name))
        cmd = f'sshpass -p "{self.target.password}" rsync --relative  -e "ssh -p {self.target.port} -o PubkeyAuthentication=yes   -o stricthostkeychecking=no" -avi {file_name} root@{self.target.ip}:{remote_file}'
        BashUtil.run_command_print_progress(cmd)
        return remote_file

    def download_file(self, remote_path, local_path):
        """
        Upload a directory

        Parameters
        ----------

        Returns
        -------

        """
        local_path = os.path.abspath(local_path)
        cmd = f'sshpass -p "{self.target.password}" rsync  -e "ssh -p {self.target.port}" -avi root@{self.target.ip}:{remote_path} {local_path} '
        BashUtil.run_command_print_progress(cmd)

    def download_dir(self, dir_name=""):
        """
        Download a directory
        Parameters
        ----------
        dir_name : the dir to download

        Returns
        -------

        """
        _r = os.path.join(self.target.home, dir_name)
        _l = os.path.join(self.source.home, dir_name)
        cmd = f'sshpass -p "{self.target.password}" rsync  -e "ssh -p {self.target.port}" -avi  root@{self.target.ip}:{_r}/ {_l}/'
        return BashUtil.run_command_print_progress(cmd)


if __name__ == '__main__':
    pass
    # remote_conf = ESC.CPU_10_219_WAN
    # ur9 = UtilRsyncV2(ESC.LOCAL_HOST, remote_conf)
    # remote_file = ur9.upload_file("calc_metric.py")
