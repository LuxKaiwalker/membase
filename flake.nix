{
  description = "Augmented Personal Memory MVP - dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      pythonEnv = pkgs.python313.withPackages (ps: with ps; [
        fastapi
        uvicorn
        sqlalchemy
        alembic
        pydantic
        pydantic-settings
        psycopg
        pgvector
        pypdf
        pillow
        httpx
        python-multipart
        jinja2
        aiofiles
        python-magic
        rapidfuzz
        tenacity
        pyyaml
        pytest
        pytest-asyncio
        pytest-cov
      ]);

      # postgresql built with the pgvector extension available to CREATE EXTENSION.
      postgresWithVector = pkgs.postgresql.withPackages (ps: [ ps.pgvector ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pythonEnv
          postgresWithVector
          pkgs.ollama
        ];

        # See CLAUDE.md "NixOS footguns" for why each of these exists.
        shellHook = ''
          # git rev-parse, not $PWD: resolves to the repo root even if
          # `nix develop` is run from a subdirectory (e.g. backend/), and
          # -- unlike $PWD -- stays correct even if MEMBASE_ROOT is read
          # later in the shell's life after a `cd` elsewhere.
          export MEMBASE_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
          export PGDATA="$MEMBASE_ROOT/.data/postgres"
          export PGHOST_DIR="$MEMBASE_ROOT/.data/postgres-socket"
          export PGPORT=5544
          export PGUSER=membase
          export PGDATABASE=membase
          export DATABASE_URL="postgresql+psycopg://$PGUSER@/$PGDATABASE?host=$PGHOST_DIR&port=$PGPORT"
          export OLLAMA_HOST="http://127.0.0.1:11434"
          export LOCALE_ARCHIVE="${pkgs.glibcLocales}/lib/locale/locale-archive"
          export UPLOADS_DIR="$MEMBASE_ROOT/.data/uploads"
          mkdir -p "$PGHOST_DIR" "$UPLOADS_DIR"
          echo "membase dev shell ready."
          echo "  scripts/dev-db.sh start   # init + start project-local postgres on port $PGPORT"
          echo "  scripts/pull-models.sh    # pull the ollama embedding model"
        '';
      };
    };
}
