(() => {
  const defaultUrl = "http://127.0.0.1:8000";
  const params = new URLSearchParams(globalThis.location.search);
  const queryUrl = params.get("api") || params.get("apiBaseUrl") || "";
  const storageUrl = globalThis.localStorage.getItem("spe_api_base_url") || "";
  const deployedHost = String(globalThis.location.hostname || "").toLowerCase();
  const isDeployed =
    deployedHost &&
    deployedHost !== "localhost" &&
    deployedHost !== "127.0.0.1";
  const fallbackUrl = isDeployed ? "" : defaultUrl;

  globalThis.SPE_API_BASE_URL =
    globalThis.SPE_API_BASE_URL || queryUrl || storageUrl || fallbackUrl;
})();
