(() => {
  const defaultUrl = "http://127.0.0.1:8000";
  const params = new URLSearchParams(globalThis.location.search);
  const queryUrl = params.get("api") || params.get("apiBaseUrl") || "";
  const storageUrl = globalThis.localStorage.getItem("spe_api_base_url") || "";

  globalThis.SPE_API_BASE_URL =
    globalThis.SPE_API_BASE_URL || queryUrl || storageUrl || defaultUrl;
})();
