"""Inspect real permissions on disposable credential files."""

import os

from tests.test_runtime_platform import _windows_acl_report


def assert_private_file(path):
    if os.name == "nt":
        report = _windows_acl_report(path)
        assert report["protected"], report
        assert report["users"] == [report["current"]], report
        assert report["rights"] == ["FullControl"], report
    else:
        assert path.stat().st_mode & 0o777 == 0o600


def allow_public_read(path):
    if os.name == "nt":
        report = _windows_acl_report(
            path,
            """
$world = [Security.Principal.SecurityIdentifier]::new('S-1-1-0')
$rule = [Security.AccessControl.FileSystemAccessRule]::new(
    $world, [Security.AccessControl.FileSystemRights]::Read,
    [Security.AccessControl.AccessControlType]::Allow)
$acl.AddAccessRule($rule)
""",
        )
        assert "S-1-1-0" in report["users"], report
    else:
        path.chmod(0o644)
