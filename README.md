# Forward Looking Tool

A frontend for the Forward Looking Tool, built with Electron, Vite and Preact.

## Running the tool

First, make sure you have Python 3 and `pip` installed and included in your PATH environment variable. You must install it using the [standalone installer](https://www.python.org/downloads/), the Windows Store version is known to cause issues with some libraries.

Download [the latest release](https://github.com/RO010000894_EYGS/ForwardLookingTool/releases/latest) of this tool. On your first run it should automatically install every missing package using pip.

## Development

### Prerequisites

Make sure you have [node.js and npm](https://nodejs.org/en/download/) installed beforehand.

```ps
# Clone the repository
git clone https://github.com/madalin312/forward_looking_tool

cd ForwardLookingTool

# Install pnpm (https://pnpm.io/)
npm install -g pnpm

# Install the dependencies
pnpm i
```

### Running a development build with hot reload

```ps
pnpm dev
```

### Building an executable

```ps
pnpm build
```

### Bumping the version

First, commit your changes:

```ps
git add -A
git commit -m "commit message here"
```

Update the package.json version field with the new version, using the major.minor.patch format (e.g. 1.2.3). Commit the new file:

```ps
git commit -m 'bump version to 1.2.3' package.json
```

Push the commits:

```ps
git push origin
```

Add a new tag `vx.y.z`:

```ps
git tag v1.2.3
```

Push the new tag (this will trigger an automated build):

```ps
git push origin --tags
```