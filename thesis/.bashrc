# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias check='squeue --me'
alias monitor='watch -n 1 tail -n 40'
alias token='cat /home/alexander.huang/repositories/huang-thesis/token.txt'
alias connect='srun --pty --partition=barton --nodelist=compute-9-31 --gres=gpu:1 --mem=8G --time=7- bash'
alias sinfo='sinfo -o "%P %G %D %N %e %t"'
function cshell () { 
    srun --pty --partition=$1 --gres=gpu:1 --mem=256G --time=72:00:00 bash
}
function nbsetup () {
    conda activate thesis 
    jupyter-notebook --port 8888 --ip ${SLURM_STEP_NODELIST}.hamming.cluster
}


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/share/spack/gcc-10.3.0/miniconda3-23.1.0-4vp/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/share/spack/gcc-10.3.0/miniconda3-23.1.0-4vp/etc/profile.d/conda.sh" ]; then
        . "/share/spack/gcc-10.3.0/miniconda3-23.1.0-4vp/etc/profile.d/conda.sh"
    else
        export PATH="/share/spack/gcc-10.3.0/miniconda3-23.1.0-4vp/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

